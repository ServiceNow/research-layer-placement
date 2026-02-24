"""Utility functions for MATH task evaluation.

Provides mathematical equivalence checking and answer extraction for MATH-500 tasks.
Based on: https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
"""

from typing import Any


def process_docs(dataset: Any) -> Any:
    """Pass through documents unchanged."""
    return dataset


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    """Check if model answer matches ground truth using mathematical equivalence.

    Extracts answer from between $ delimiters if present, then compares
    using normalized string matching.
    """
    retval = 0
    # Extract answer from between $ delimiters
    indices = [pos for pos, char in enumerate(results[0]) if char == "$"]
    if len(indices) <= 1:
        answer = results[0]
    else:
        answer = results[0][indices[0] + 1 : indices[-1]]

    # Also try extracting from [BEGIN FINAL RESPONSE]...<|end|> block
    import re

    final_response_match = re.search(
        r"\[BEGIN FINAL RESPONSE\]\s*(.+?)\s*<\|end\|>", results[0], re.DOTALL
    )
    if final_response_match:
        answer = final_response_match.group(1).strip()
        # Strip $ delimiters if present
        if answer.startswith("$") and answer.endswith("$"):
            answer = answer[1:-1]
        # Strip \[ \] delimiters if present
        if answer.startswith("\\[") and answer.endswith("\\]"):
            answer = answer[2:-2].strip()

    ground_truth = remove_boxed(last_boxed_only_string(doc["solution"]))
    if is_equiv(answer, ground_truth):
        retval = 1

    return {
        "exact_match": retval,
    }


def is_equiv(str1: str | None, str2: str | None, verbose: bool = False) -> bool:
    """Check if two mathematical expressions are equivalent after normalization."""
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s: str | None) -> str | None:
    """Extract content from \\boxed{...} or \\boxed ... format."""
    if s is None:
        return None

    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] == left:
            return s[len(left) :]

    left = "\\boxed{"

    if not s.startswith(left):
        return s
    if not s.endswith("}"):
        return s

    return s[len(left) : -1]


def last_boxed_only_string(string: str | None) -> str | None:
    """Find the last \\boxed{...} in the string and return it."""
    if string is None:
        return None

    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string: str) -> str:
    """Normalize fraction notation like \\frac12 to \\frac{1}{2}."""
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string: str) -> str:
    """Convert a/b notation to \\frac{a}{b} for integers."""
    if len(string.split("/")) != 2:
        return string
    a_str = string.split("/")[0]
    b_str = string.split("/")[1]
    try:
        a = int(a_str)
        b = int(b_str)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except (ValueError, AssertionError):
        return string


def remove_right_units(string: str) -> str:
    """Remove unit text from the right side of an expression."""
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string: str) -> str:
    """Normalize \\sqrt3 to \\sqrt{3}."""
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string: str) -> str:
    """Normalize a mathematical string for comparison.

    Applies various normalizations:
    - Removes spaces, newlines
    - Normalizes fractions, square roots
    - Removes units
    - Handles tfrac/dfrac variants
    """
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Exposed
    # by problem 12 of hexagonal
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "x = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc.
    # Even works with \frac1determine because they have digits
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y://teleedelta args://teleedelta{X}{Y} is incorrect
    string = fix_a_slash_b(string)

    return string
