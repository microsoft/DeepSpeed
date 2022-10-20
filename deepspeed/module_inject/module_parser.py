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
                elif method.name == "forward":
                    forward_source = ast.get_source_segment(file_content, method)
            #stop searching class list
            break
    return init_source, forward_source

                    

def get_linear_layers(source):
    matches = re.findall(r"self.(.*?) = nn.Linear", source)
    if not matches:
        #check for attributes at block class level     
        matches = re.findall(r"self.(.*?).append\((.*?)\(", source)
        if not matches:
            #check for attributes at layer class level
            matches = re.findall(r"self.(.*?) = (\w*?)\(", source)
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
    
    #if need_all_reduce is True we do not need to check again
    i_need_all_reduce = need_all_reduce
    
    for name, attribute in matches:
        #get source for all methods in match list
        init_source, forward_source = get_class_source(node, attribute)
        if init_source != "" and forward_source != "":
            #check for linear layers
            result, i_matches = get_linear_layers(init_source)
            if not need_all_reduce:
                #search for functions (LayerNorm, GroupNorm, etc) that need all reduce
                i_need_all_reduce = check_layer_norm(forward_source) | check_layer_norm(init_source)
            if result and (need_all_reduce or i_need_all_reduce):               
                #add linear layers to list   
                i_matches = update_name_list(name, i_matches)
                linear_layer_list.append(i_matches)
                #reset i_need_all_reduce
                i_need_all_reduce = False
            if not result:
                #add next level of class methods to check 
                new_matches = new_matches + i_matches
    return i_need_all_reduce, new_matches


if __name__ == "__main__":
    linear_layer_list = []
    injection_policy_list = []
    need_all_reduce = False

    #parse file as abstract syntax tree
    with open(args.file, "r") as f:
        file_content = f.read()
        node = ast.parse(file_content)

    #get source code of specified module
    init_source, forward_source = get_class_source(node, args.module)

    if init_source != "" and forward_source != "":
        #search for functions (LayerNorm, GroupNorm, etc) that need all reduce
        need_all_reduce = check_layer_norm(forward_source) | check_layer_norm(init_source)

        #check for linear layers in source code        
        result, matches = get_linear_layers(init_source)
         
        if result & need_all_reduce:
            #add linear layers to list
            update_name_list(name, matches)
            linear_layer_list.append(matches)
        else:
            while len(matches):
                need_all_reduce, matches = check_matches(need_all_reduce, matches)

    #generate injection policy gems from name and linear layer lists
    for group in linear_layer_list:
        injection_policy_list.append(group[-1])
    
    #remove duplicate names. All gems with same parent.name are all-reduced
    injection_policy_list = set(injection_policy_list)
    
    #print injection policy
    injection_policy = {}
    injection_policy.update({args.module: tuple(injection_policy_list)})
    print("injection_policy={" + args.module + ": " + str(tuple(injection_policy_list)) + "}")
    
