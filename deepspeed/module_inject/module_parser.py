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


#def get_linear_layers(file_content):
#    matches = re.findall(r"self.(.*?) = nn.Linear(.+?)", file_content)

if __name__ == "__main__":
    #parse file for specified module
    with open(args.file, "r") as f:
        file_content = f.read()
        node = ast.parse(file_content)

    #functions = [n for n in node.body if isinstance(n, ast.FunctionDef)]
    classes = [n for n in node.body if isinstance(n, ast.ClassDef)]

    #for function in functions:
    #    show_info(function)

    for class_ in classes:
        if class_.name == args.module:
            print("Class name:", class_.name)
            methods = [n for n in class_.body if isinstance(n, ast.FunctionDef)]
            for method in methods:
                if method.name == "__init__":
                    #show_info(method)
                    print(ast.get_source_segment(file_content, method))



    #inside module, look for "self.<name1> = <submodle>(...)"
        #in submodule, parse for "self.<name2> = nn.Linear(...)"
        #the last match is used to generate injection policy name1.name2

    #else look for "self.layer.append(<layermodule>(...))"
        #In layermodule, look for "self.<name1> = submodule(...)"
            #in submodule, parse for "self.<name2> = nn.Linear(...)"
            #the last match is used to generate injection policy name1.name2

    #more generic approach:
    #always look for linear layer first, if none look for submodules and go level deeper
