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

def show_info(functionNode):
    print("Function name:", functionNode.name)
    print("Args:")
    for arg in functionNode.args.args: 
        print("\tParameter name:", arg.arg)

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
        print(f"No linear layers found, checking attributes...")
        matches = re.findall(r"self.(.*?) = (.*?)\(", method_source)
        print("attributes: ", matches)     
        return False, matches
    else:
        print("linear layers: ", matches)
        return True, matches        


if __name__ == "__main__":
    linear_layer_list = []

    #parse file for specified module
    with open(args.file, "r") as f:
        file_content = f.read()
        node = ast.parse(file_content)

    #get source code of top class
    source = get_class_source(node, args.module)

    #check for self.layer.append case
    #layer_matches = re.findall(r"self.layer.append(.*?)\(", source)
    #if layer_matches:
    #    for layer in layer_matches:
    #        source = get_class_source(node, layer)
    #        result, matches = get_linear_layers(source)
            
    
    result, matches = get_linear_layers(source)

    print("result: ", result)
    print("matches: ", matches)

    if not result:
        for name, attribute in matches:
            source = get_class_source(node, attribute)
            if source is not None:
                intermediate_result, i_matches = get_linear_layers(source)
                if intermediate_result:
                    linear_layer_list.append(i_matches)
                if not intermediate_result:
                    print("matches: ", matches)
                    matches = matches + i_matches
                    print("i_mathces: ", matches)

    print("FINAL RESULT: ", linear_layer_list)
            
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
