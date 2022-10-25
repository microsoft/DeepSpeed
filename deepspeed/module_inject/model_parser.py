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
args = parser.parse_args()




if __name__ == "__main__":
    module_list = []

    #parse file as abstract syntax tree
    with open(args.file, "r") as f:
        file_content = f.read()
        node = ast.parse(file_content)

    classes = [n for n in node.body if isinstance(n, ast.ClassDef)]
    for class_ in classes:
        print("Class name:", class_.name)
        module_list.append(class_.name)
        match = re.search("Layer", class_.name)
        if match is not None:
            print("Match")
        #methods = [n for n in class_.body if isinstance(n, ast.FunctionDef)]
        #for method in methods:
        #    if method.name == "__init__":
        #        source = ast.get_source_segment(file_content, method)

    print('(', end="")
    for module in module_list:
        print('"' + module + '" ', end="")
    print(')')

