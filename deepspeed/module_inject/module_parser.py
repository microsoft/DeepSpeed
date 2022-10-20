import os
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
        "--module",
        "-m",
        type=str,
        default="",
        help="target module inside model file",
        )
args = parser.parse_args()


def get_class_source(node, class_name):
    init_source = ""
    forward_source = ""
    classes = [n for n in node.body if isinstance(n, ast.ClassDef)]
    for class_ in classes:
        if class_.name == class_name:
            #print("Class name:", class_.name)
            methods = [n for n in class_.body if isinstance(n, ast.FunctionDef)]
            for method in methods:
                if method.name == "__init__":
                    init_source = ast.get_source_segment(file_content, method)
                    #return method_source
                elif method.name == "forward":
                    forward_source = ast.get_source_segment(file_content, method)
            #stop searching class list
            break
    return init_source, forward_source

                    

def get_linear_layers(method_source):
    matches = re.findall(r"self.(.*?) = nn.Linear", method_source)
    if not matches:
        #check for attributes at block class level     
        matches = re.findall(r"self.(.*?).append\((.*?)\(", method_source)
        if not matches:
            #check for attributes at layer class level
            matches = re.findall(r"self.(.*?) = (\w*?)\(", method_source)
        return False, matches
    else:
        return True, matches        


def check_layer_norm(source):
    match = re.search(r"LayerNorm|layer_norm", source)
    if match is not None:
        return True
    else:
        return False


def update_name_list(name, matches):
    new_list = []
    for match in matches:    
        new_list = new_list + [name + "." + match] 
    return new_list


def check_matches(need_all_reduce, matches):
    new_matches = []
    i_need_all_reduce = need_all_reduce
    for name, attribute in matches:
        init_source, forward_source = get_class_source(node, attribute)
        if init_source != "" and forward_source != "":
            result, i_matches = get_linear_layers(init_source)
            #print("i_matches: ", i_matches)
            #print("result: ", result)
            #print("need_all_reduce: ", need_all_reduce)
            if not need_all_reduce:
                i_need_all_reduce = check_layer_norm(forward_source) | check_layer_norm(init_source)
            if result and (need_all_reduce or i_need_all_reduce):               
                #add linear layers to list   
                i_matches = update_name_list(name, i_matches)
                linear_layer_list.append(i_matches)
                i_need_all_reduce = False
            if not result:
                #add next level of class methods to check 
                new_matches = new_matches + i_matches
    return i_need_all_reduce, new_matches


if __name__ == "__main__":
    linear_layer_list = []
    injection_policy_list = []
    all_reduce_list = []
    need_all_reduce = False 
    #parse file for specified module
    with open(args.file, "r") as f:
        file_content = f.read()
        node = ast.parse(file_content)

    #get source code of top class
    init_source, forward_source = get_class_source(node, args.module)

    if init_source != "" and forward_source != "":
        need_all_reduce = check_layer_norm(forward_source) | check_layer_norm(init_source)

        #check for linear layers in source code        
        result, matches = get_linear_layers(init_source)
         
        if result & need_all_reduce:
            #add linear layers to list
            update_name_list(name, matches)
            linear_layer_list.append(matches)
        else:
            while len(matches):
                #print("checking matches...", matches)
                need_all_reduce, matches = check_matches(need_all_reduce, matches)

        #if no linear layers found, check attribute source code
        #if not result:
        #    while len(matches):
        #        #print("checking matches...", matches)
        #        need_all_reduce, matches = check_matches(need_all_reduce, matches)
        #elif need_all_reduce:
        #    #add linear layers to list
        #    update_name_list(name, matches)
        #    linear_layer_list.append(matches)


    #generate injection policy gems from name and linear layer lists
    for group in linear_layer_list:
        injection_policy_list.append(group[-1])
    
    #remove duplicate names. All gems with same parent.name are all-reduced
    injection_policy_list = set(injection_policy_list)
    
    #print injection policy
    injection_policy = {}
    injection_policy.update({args.module: tuple(injection_policy_list)})
    print("injection_policy={" + args.module + ": " + str(tuple(injection_policy_list)) + "}")
    
