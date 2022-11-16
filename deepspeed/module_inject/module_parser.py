import re
import ast
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file",
    "-f",
    type=str,
    default="",
    help="filepath of model file to be parsed",
)
parser.add_argument(
    "--output_file",
    "-o",
    default="./module_parser_output.csv",
    help="output file to write parser results to",
)
args = parser.parse_args()


def get_class_source(node, class_name):
    init_source = ""
    forward_source = ""
    classes = [n for n in node.body if isinstance(n, ast.ClassDef)]
    for class_ in classes:
        if class_.name == class_name:
            methods = [n for n in class_.body if isinstance(n, ast.FunctionDef)]
            for method in methods:
                if method.name == "__init__":
                    init_source = ast.get_source_segment(file_content, method)
                elif method.name == "forward":
                    forward_source = ast.get_source_segment(file_content, method)
            #stop searching class list
            break
    return init_source, forward_source


def get_module(node):
    module_list = []

    classes = [n for n in node.body if isinstance(n, ast.ClassDef)]
    for class_ in classes:
        methods = [n for n in class_.body if isinstance(n, ast.FunctionDef)]
        for method in methods:
            if method.name == "__init__":
                source = ast.get_source_segment(file_content, method)
                match = re.search(r"nn.ModuleList\(\s*\[\s*(.*?)\(\s*\w*config(\)|,|.)",
                                  source)
                if match is not None:
                    module_list = module_list + [match.group(1)]
    if module_list is not None:
        return list(set(module_list))
    else:
        print("No modules found")
        return None


def get_linear_layers(source):
    linear_matches = re.findall(r"self.(.*?) = nn.(Linear)", source)

    #check for attributes at block class level
    block_matches = re.findall(r"self.(.*?).append\((.*?)\(", source)

    #check for attributes at layer class level
    layer_matches = re.findall(r"self.(.*?) = (\w*?)\(", source)

    matches = block_matches + layer_matches

    if not linear_matches:
        return False, linear_matches, matches
    else:
        return True, linear_matches, matches


def check_layer_norm(source):
    match = re.search(r"LayerNorm|layer_norm", source)
    if match is not None:
        return True
    else:
        return False


def update_name_list(parent_name, matches):
    new_list = []
    for match, linear in matches:
        if parent_name == "self":
            new_list = new_list + ["." + match]
        else:
            new_list = new_list + [parent_name + "." + match]
    return new_list


def check_matches(need_all_reduce, matches):
    new_matches = []

    #if need_all_reduce is True we do not need to check again
    i_need_all_reduce = need_all_reduce

    for name, attribute in matches:
        #get source for all methods in match list
        init_source, forward_source = get_class_source(node, attribute)
        if init_source != "" and forward_source != "":
            #check for linear layers
            result, linear_matches, i_matches = get_linear_layers(init_source)
            if not need_all_reduce:
                #search for functions (LayerNorm, GroupNorm, etc) that need all reduce
                i_need_all_reduce = check_layer_norm(forward_source) | check_layer_norm(
                    init_source)
            if result and (need_all_reduce or i_need_all_reduce):
                #add linear layers to list
                linear_matches = update_name_list(name, linear_matches)
                linear_layer_list.append(linear_matches)
                #reset i_need_all_reduce
                i_need_all_reduce = False
            if i_matches:
                #add next level of class methods to check
                new_matches = new_matches + i_matches
    return i_need_all_reduce, new_matches


def get_key_name():
    model_name = re.search(r"modeling_(.*?).py", args.file)
    #remove underscore characters
    key = re.sub('_', '', model_name.group(1))
    return key


if __name__ == "__main__":

    modules_policy_list = []

    #parse file as abstract syntax tree
    with open(args.file, "r") as f:
        file_content = f.read()
        node = ast.parse(file_content)

    key = get_key_name()
    modules = get_module(node)

    for module in modules:
        linear_layer_list = []
        injection_policy_list = []
        need_all_reduce = False

        #get source code of specified module
        init_source, forward_source = get_class_source(node, module)

        if init_source != "" and forward_source != "":
            #search for functions (LayerNorm, GroupNorm, etc) that need all reduce
            need_all_reduce = check_layer_norm(forward_source) | check_layer_norm(
                init_source)

            #check for linear layers in source code
            result, linear_matches, matches = get_linear_layers(init_source)

            if result & need_all_reduce:
                #add linear layers to list
                linear_matches = update_name_list("self", linear_matches)
                linear_layer_list.append(linear_matches)
            if matches is not None:
                while len(matches):
                    need_all_reduce, matches = check_matches(need_all_reduce, matches)

        #generate injection policy gems from name and linear layer lists
        for group in linear_layer_list:
            injection_policy_list.append(group[-1])

        #remove duplicate names. All gems with same parent.name are all-reduced
        injection_policy_list = list(set(injection_policy_list))
        if len(injection_policy_list):
            injection_policy_list = [module, injection_policy_list]
            modules_policy_list.append(injection_policy_list)

    #print injection policy
    #injection_policy = {}
    #injection_policy.update({args.module: tuple(injection_policy_list)})
    #print("injection_policy={" + args.module + ": " + str(tuple(injection_policy_list)) + "}")

    #write results to output file
    if len(modules_policy_list):
        print(modules_policy_list)
        output_string = key + "=dict("
        for module, injection_policy in modules_policy_list:
            output_string = output_string + module + "=("
            for gem in injection_policy:
                output_string = output_string + '"' + gem + '", '
            output_string = output_string + "), "
        output_string = output_string + "),"
        ofile = open(args.output_file, "a")
        ofile.write(output_string + '\n')
        ofile.close()
    else:
        print(injection_policy_list)
        print("no policy for ", key)
