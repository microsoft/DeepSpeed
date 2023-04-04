import os
import argparse

PY_NEW_HEADER = """# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""

CPP_NEW_HEADER = """// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team
"""

PY_SL_COMMENT = "#"
PY_ML_SINGLE = "'''"
PY_ML_DOUBLE = '"""'

CPP_SL_COMMENT = "//"
CPP_ML_OPEN = "/*"
CPP_ML_CLOSE = "*/"

DELIM = "|/-\|/-\|BARRIER|/-\|/-\|"  # noqa: W605


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", type=str, help="Repository directory")
    #parser.add_argument("--file_ext", type=str, nargs="+", default=[".py",
    #                ".c",
    #                ".cpp",
    #                ".cu",
    #                ".h",
    #                ".hpp",
    #                ".cuh",
    #                ".cc",
    #                ".hip",
    #                ".tr",
    #            ],
    #            choices=[   ".py",
    #                ".c",
    #                ".cpp",
    #                ".cu",
    #                ".h",
    #                ".hpp",
    #                ".cuh",
    #                ".cc",
    #                ".hip",
    #                ".tr",
    #            ],
    #            help="File extensions to process")
    args = parser.parse_args()
    return args


def get_header_py(fp):
    with open(fp, "r") as f:
        lines = iter(l for l in f.readlines())

    header = []
    rest = []
    in_multiline = False
    multiline_type = None

    while (l := next(lines, None)) is not None:
        l = l.strip()
        if l.startswith(PY_ML_SINGLE) or l.startswith(PY_ML_DOUBLE):
            # Detected multiline comment
            if in_multiline and multiline_type == l[:3]:
                # Ended a multiline comment
                in_multiline = False
            else:
                # Started a multiline comment
                in_multiline = True
                multiline_type = l[:3]
            if l.endswith(multiline_type) and len(l) >= 6:
                # Opened and closed multiline comment on single line
                in_multiline = False
        elif in_multiline and l.endswith(multiline_type):
            # Ended a multiline comment
            in_multiline = False
        elif not (in_multiline or l.startswith(PY_SL_COMMENT) or l == ""):
            # Not in a comment
            rest = [l + "\n"] + list(lines)
            break
        header.append(l)

    if rest == []:
        rest = list(lines)

    #return "\n".join(header), "".join(rest)
    return header, rest


def get_header_cpp(fp):
    with open(fp, "r") as f:
        lines = iter(l for l in f.readlines())

    header = []
    rest = []
    in_multiline = False
    multiline_type = None

    while (l := next(lines, None)) is not None:
        l = l.strip()
        if l.startswith(CPP_ML_OPEN):
            # Detected multiline comment
            if not l.endswith(CPP_ML_CLOSE):
                # multiline comment not closed on same line
                in_multiline = True
        elif l.endswith(CPP_ML_CLOSE):
            # Ended a multline comment
            in_multiline = False
        elif not in_multiline or l.startswith(CPP_SL_COMMENT) or l.isspace():
            # Not in a comment
            rest = [l + "\n"] + list(lines)
            break
        header.append(l)

    if rest == []:
        rest = list(lines)

    #return "\n".join(header), "".join(rest)
    return header, rest


def main(args):
    preserve_text = {}
    for root, dirs, fnames in os.walk(args.repo_dir):
        # Walk across directory looking for all python files
        fpaths = [os.path.join(root, fn) for fn in fnames if fn.endswith(".py")]
        for fp in fpaths:
            file_header, rest_of_file = get_header_py(fp)
            header_text = "\n".join(file_header)
            if header_text.strip() == "":
                pass
            elif header_text in preserve_text:
                pass
            else:
                print("\n", DELIM, "\n")
                for idx, line in enumerate(file_header):
                    print(f"{idx}: {line}")
                keep_idx = input(
                    "Please indicate the FIRST line of the Header to KEEP. Or press ENTER if there are no lines to keep: "
                )
                if keep_idx == "":
                    preserve_text[header_text] = ""
                else:
                    preserve_text[header_text] = file_header[int(keep_idx):]

            if header_text.strip() == "":
                text_to_preserve = ""
            else:
                text_to_preserve = preserve_text[header_text]
                new_header = PY_NEW_HEADER
                shebangs = [l for l in text_to_preserve if l.startswith("#!")]
                text_to_preserve = [
                    l.replace(PY_ML_SINGLE, "").replace(PY_ML_DOUBLE, "").replace(PY_SL_COMMENT, "").strip()
                    for l in text_to_preserve if l not in shebangs
                ]
                text_to_preserve = [l for l in text_to_preserve if l != ""]
                if "".join(text_to_preserve) != "":
                    text_to_preserve = '\n"""\n' + "\n".join(text_to_preserve) + '\n"""'
                else:
                    text_to_preserve = ""
            if shebangs:
                shebang = "\n".join(shebangs) + "\n"
            else:
                shebang = ""
            new_header = shebang + PY_NEW_HEADER + text_to_preserve

            new_file_contents = new_header + "\n" + "".join(rest_of_file)
            with open(fp, "w") as f:
                f.write(new_file_contents)

    for root, dirs, fnames in os.walk(args.repo_dir):
        fpaths = [
            os.path.join(root, fn) for fn in fnames if any(
                fn.endswith(ext) for ext in [
                    ".c",
                    ".cpp",
                    ".cu",
                    ".h",
                    ".hpp",
                    ".cuh",
                    ".cc",
                    ".hip",
                    ".tr",
                ])
        ]
        for fp in fpaths:
            file_header, rest_of_file = get_header_cpp(fp)
            header_text = "\n".join(file_header)
            if header_text.strip() == "":
                pass
            elif header_text in preserve_text:
                pass
            else:
                print("\n", DELIM, "\n")
                for idx, line in enumerate(file_header):
                    print(f"{idx}: {line}")
                keep_idx = input(
                    "Please indicate the FIRST line of the Header to KEEP. Or press ENTER if there are no lines to keep: "
                )
                if keep_idx == "":
                    preserve_text[header_text] = ""
                else:
                    preserve_text[header_text] = file_header[int(keep_idx):]

            if header_text.strip() == "":
                text_to_preserve = ""
            else:
                text_to_preserve = preserve_text[header_text]
                new_header = CPP_NEW_HEADER
                text_to_preserve = [
                    l.replace(CPP_ML_OPEN, "").replace(CPP_ML_CLOSE, "").replace(CPP_SL_COMMENT, "").strip()
                    for l in text_to_preserve
                ]
                text_to_preserve = [l for l in text_to_preserve if l != ""]
                if "".join(text_to_preserve) != "":
                    text_to_preserve = '\n"""\n' + "\n".join(text_to_preserve) + '\n"""'
                else:
                    text_to_preserve = ""
            new_header = CPP_NEW_HEADER + text_to_preserve

            new_file_contents = new_header + "\n" + "".join(rest_of_file)
            with open(fp, "w") as f:
                f.write(new_file_contents)


if __name__ == "__main__":
    args = parser_args()
    main(args)
