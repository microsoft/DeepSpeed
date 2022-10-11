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
    classes = [n for n in node.body if isinstance(n, ast.ClassDef)]

    for class_ in classes:
        if class_.name == class_name:
            print("Class name:", class_.name)
            methods = [n for n in class_.body if isinstance(n, ast.FunctionDef)]
            for method in methods:
                if method.name == "__init__":
                    method_source = ast.get_source_segment(file_content, method)
                    return method_source
                    

def get_linear_layers(method_source):
    matches = re.findall(r"self.(.*?) = nn.Linear", method_source)
    if not matches:
        #check for attributes at layer class level
        #print("No linear layers found, checking attributes...")
        #matches = re.findall(r"self.(.*?) = (.*?)\(", method_source)
        matches = re.findall(r"self.(.*?).append\((.*?)\(", method_source)
        if not matches:
            #check for attributes at block class level
            #print("Checking layer class level...")
            #matches = re.findall(r"self.(.*?).append\((.*?)\(", method_source)
            matches = re.findall(r"self.(.*?) = (\w*?)\(", method_source)
            #if not matches:
                #print("Warning: No matches found")
        return False, matches
    else:
        return True, matches        


def update_name_list(name, matches):
    new_list = []
    for match in matches:    
        new_list = new_list + [name + "." + match] 
    return new_list

def update_name_list_2(name, matches):
    new_list = []
    for match_name, match_attribute in matches:
        new_list.append(tuple([name + "." + match_name, match_attribute]))
        print("HERE", new_list)
    return new_list

def check_matches(matches):
    new_matches = []
    for name, attribute in matches:
        source = get_class_source(node, attribute)
        if source is not None:
            result, i_matches = get_linear_layers(source)
            if result:
                #add linear layers to list
                name_list.append(name)
                i_matches = update_name_list(name, i_matches)
                linear_layer_list.append(i_matches)
            if not result:
                #add next level of class methods to check
                name_list.append(name)
                i_matches = update_name_list_2(name, i_matches) 
                new_matches = new_matches + i_matches
    return new_matches


if __name__ == "__main__":
    name_list = []
    linear_layer_list = []

    #parse file for specified module
    with open(args.file, "r") as f:
        file_content = f.read()
        node = ast.parse(file_content)

    #get source code of top class
    source = get_class_source(node, args.module)

    #check for linear layers in source code        
    result, matches = get_linear_layers(source)
    

    #if no linear layers found, check attribute source code
    if not result:
        while len(matches):
            print("checking matches...", matches)
            matches = check_matches(matches)

    else:
        #add linear layers to list
        update_name_list(name, matches)
        linear_layer_list.append(matches)

    #generate injection policy gems from name and linear layer lists
    print(name_list)
    print(linear_layer_list)

    for group in linear_layer_list:
        print(group[-1])
            
    #functions = [n for n in node.body if isinstance(n, ast.FunctionDef)]
    #classes = [n for n in node.body if isinstance(n, ast.ClassDef)]

    #for function in functions:
    #    show_info(function)

    #for class_ in classes:
    #    if class_.name == args.module:
    #        print("Class name:", class_.name)
    #        methods = [n for n in class_.body if isinstance(n, ast.FunctionDef)]
    #        for method in methods:
    #            if method.name == "__init__":
                    #show_info(method)
    #                method_source = ast.get_source_segment(file_content, method)
    #                print(method_source)
    #                get_linear_layers(class_.name, method_source)



    #inside module, look for "self.<name1> = <submodle>(...)"
        #in submodule, parse for "self.<name2> = nn.Linear(...)"
        #the last match is used to generate injection policy name1.name2

    #else look for "self.layer.append(<layermodule>(...))"
        #In layermodule, look for "self.<name1> = submodule(...)"
            #in submodule, parse for "self.<name2> = nn.Linear(...)"
            #the last match is used to generate injection policy name1.name2

    #more generic approach:
    #always look for linear layer first, if none look for submodules and go level deeper
