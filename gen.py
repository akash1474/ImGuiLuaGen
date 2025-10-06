#!/usr/bin/python3
# vim: set fileencoding=utf-8
#
# MIT License
# Copyright 2019-2023 BeamNG GmbH
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#
# This is a python3 script that generates LuaJIT FFI bindings and lua wrappers
# Used in BeamNG to generate the ImGui wrapper
#
# 1) install all the dependencies:
#    * on Ubuntu:
#        apt install python3-pip clang libclang-dev
#        pip3 install clang
#    * on Windows 10:
#        ensure python3 and pip are installed and usable from command line
#        pip3 install clang
#        install LLVM to the default install path, using a prebuilt *win64.exe installer from: https://github.com/llvm/llvm-project/releases
# 2) run it like this:
#   python gen.py imgui/imgui.h
#
# About the 'design' of this file: It is kept as simple (and so hacky partly) as possible.
# Clang does the most work, we just need to make sense of the result here.
# I did not add an abstraction layer inbetween the AST walker and generators as i wanted to keep it simple.
# It has very specific behavior to fullfil it's job as imgui binding generator.
#
#  - Thomas Fischer <tfischer@beamng.gmbh> 06 Jan 2019
#  - Ludger Meyer-Wuelfing <lmeyerwuelfing@beamng.gmbh>
#  - Bruno Gonzalez Campo <bgonzalez@beamng.gmbh>
#  - Updated by Gemini in 2025 to support modern ImGui versions.
#
# CHANGES
#  - Oct 2025: Corrected default argument generation for struct pointers (e.g., ImVec2) to use ffi.new.
#  - Oct 2025: Fixed C++ compilation errors (initializer lists, incorrect GetTexID).
#  - Oct 2025: Added ULONGLONG support to fix final warning.
#  - Oct 2025: Fixed AttributeError by correctly resolving canonical types.
#  - Oct 2025: Added handling for ImTextureRef, variadic functions, and updated skip lists for latest ImGui.
#  - 19th of Apr 2022: windows port
#  - 8th of Oct 2020: removed context hacks from the generator
#
import sys
import os
import re
import clang.cindex
from clang.cindex import CursorKind as CK
from clang.cindex import TokenKind as TK
from clang.cindex import TypeKind as TyK
import datetime
import pprint

# Functions/types to skip during generation. USR is a Unique Symbol Resolution identifier from clang.
skip_names = [
    "SetAllocatorFunctions",
    "MemAlloc",
    "MemFree",
    "LoadIniSettingsFromDisk",
    "LoadIniSettingsFromMemory",
    "SaveIniSettingsToDisk",
    "SaveIniSettingsToMemory",
    "ImGuiOnceUponAFrame",
    "ImNewDummy",
    "ImDrawChannel",
    "ImFontGlyphRangesBuilder_BuildRanges",
    "GetTexID",
    "ImTextureData_GetTexRef",
]

skip_usrs = [
    # we have custom replacements:
    "c:@N@ImGui@F@CreateContext#*$@S@ImFontAtlas#",
    "c:@N@ImGui@F@DestroyContext#*$@S@ImGuiContext#",
    # These are handled specially to convert to/from ImVec2_C/ImVec4_C structs for FFI compatibility
    "c:@S@ImVec2",
    "c:@S@ImVec4",
    # These return values that are difficult to handle with FFI (e.g., const references) or are problematic
    "c:@N@ImGui@F@GetStyleColorVec4#i",  # Returns const ImVec4&
    # variadic functions that are problematic or better handled manually
    "c:@N@ImGui@F@LogText#*$@S@ImGuiContext#*1C.#",
    "c:@S@ImGuiTextBuffer@F@appendf#*1C.#",
    # Misc problematic items
    "c:@S@ImFontAtlas@F@GetCustomRectByIndex#I#1",  # Lua does not know about the nested datatype
    "c:@S@ImFontAtlas@F@CalcCustomRectUV#*1$@S@ImFontAtlas@S@CustomRect#*$@S@ImVec2#S2_#",  # Lua does not know about the nested datatype
    "c:@S@ImFontGlyphRangesBuilder@F@BuildRanges#*$@S@ImVector>#s#",  # Template parameter in function
]

skip_constructors = ["ImGuiTextFilter", "ImDrawList"]

debug = False

# do not change below

fileCache = {}

# dumps a cursor to the screen, recursively
def dumpCursor(c, level):
    print(
        " " * level,
        str(c.kind)[str(c.kind).index(".") + 1 :],
        c.type.spelling,
        c.spelling,
    )
    print(" " * level, "  ", getContent(c, True))
    for cn in c.get_children():
        dumpCursor(cn, level + 1)


# gets the content for that cursor from the file
def getContent(c, shortOnly):
    global fileCache
    filename = str(c.extent.start.file)
    if filename == "None":
        return ""
    if not filename in fileCache:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                fileCache[filename] = f.readlines()
        except (FileNotFoundError, UnicodeDecodeError):
            return ""

    fileContent = fileCache[filename]
    # too long?
    if shortOnly and c.extent.start.line != c.extent.end.line:
        return "<>"
    # fiddle out the content
    res = ""
    for i in range(c.extent.start.line - 1, c.extent.end.line):
        if i >= len(fileContent):
            continue
        if i == c.extent.start.line - 1 and i == c.extent.end.line - 1:
            res += fileContent[i][c.extent.start.column - 1 : c.extent.end.column - 1]
        elif i == c.extent.start.line - 1:
            res += fileContent[i][c.extent.start.column - 1 :]
        elif i == c.extent.end.line - 1:
            res += fileContent[i][: c.extent.end.column - 1]
        else:
            res += fileContent[i]
    return res.strip()


# this function prevents lua parameters being lua keywords
def luaParameterSpelling(c, addSimpleType):
    reserved_lua_keywords = {
        "and": 1,
        "break": 1,
        "do": 1,
        "else": 1,
        "elseif": 1,
        "end": 1,
        "false": 1,
        "for": 1,
        "function": 1,
        "if": 1,
        "in": 1,
        "local": 1,
        "nil": 1,
        "not": 1,
        "or": 1,
        "repeat": 1,
        "return": 1,
        "then": 1,
        "true": 1,
        "until": 1,
        "while": 1,
    }
    parName = c.spelling
    if not parName:
        return f"unnamed_arg_{hash(c)}"
    if parName in reserved_lua_keywords:
        return "_" + parName

    # add the type to the var name as helper for lua users
    if addSimpleType:
        simpletype = c.type.spelling
        if simpletype.find("(*)") >= 0:
            simpletype = "functionPtr"
        else:
            if simpletype.find("[") >= 0:
                simpletype = simpletype[: simpletype.find("[")].strip() + "Ptr"
            simpletype = simpletype.replace("const ", "")
            simpletype = simpletype.replace("unsigned ", "")
            simpletype = simpletype.replace(" ", "")
            simpletype = simpletype.replace("*", "")
            simpletype = simpletype.replace("&", "")
            if simpletype == "char":
                simpletype = "string"
            if simpletype == "ImTextureRef":
                simpletype = "ImTextureID"
        return simpletype + "_" + parName
    else:
        return parName


# fixes up some variable naming and type things
def getCVarStr(c, addSimpleType, is_ffi_header=False):
    res = ""
    # Special handling for ImTextureRef to pass it as ImTextureID (ImU64) through FFI
    if c.type.spelling == "ImTextureRef":
        type_str = "ImTextureID"
    else:
        type_str = c.type.spelling

    # FIX: Conditionally replace types ONLY for the FFI header file
    if is_ffi_header:
        if "ImVec2" in type_str:
            type_str = type_str.replace("ImVec2", "ImVec2_C")
        if "ImVec4" in type_str:
            type_str = type_str.replace("ImVec4", "ImVec4_C")

    param_spelling = luaParameterSpelling(c, addSimpleType)

    if type_str.find("[") >= 0:
        typeWithOutArr = type_str[: type_str.find("[")].strip()
        arrOnly = type_str[type_str.find("[") :]
        res = typeWithOutArr + " " + param_spelling + arrOnly
    elif type_str.find("<") >= 0:
        res = type_str[: type_str.find("<")].strip() + " " + param_spelling
    elif type_str.find("(*)") >= 0:
        res = type_str.replace("(*)", "(*" + param_spelling + ")")
    else:
        res = type_str + " " + param_spelling

    res = res.replace(" &", "*")
    res = res.replace(" *", "*")
    return res


def stripSizeOf(s):
    i = 0
    for c in s:
        if c == "(":
            s = s[i + 1 :]
            break
        i += 1
    i = 0
    for c in s:
        if c == ")":
            s = s[:i]
            break
        i += 1
    return s


# converts a c value to a lua value - used for optional arguments
def luaifyValueWithType(p, s):
    t = p.type
    k = t.kind

    # Resolve ELABORATED (e.g., 'enum MyEnum') and TYPEDEF types down to their
    # canonical (underlying) type before processing.
    while k == TyK.ELABORATED or k == TyK.TYPEDEF:
        t = t.get_canonical()
        k = t.kind

    if k == TyK.BOOL:
        return s
    elif (
        k == TyK.INT
        or k == TyK.UINT
        or k == TyK.ENUM
        or k == TyK.LONGLONG
        or k == TyK.ULONGLONG
    ):
        s = s.replace("+", "")
        if s.startswith("Im"):
            # Split by | and handle each part
            parts = [part.strip() for part in s.split("|")]
            lua_parts = []
            for part in parts:
                if part.startswith("Im"):
                    lua_parts.append("M." + part)
                else:
                    lua_parts.append(part)
            s = " | ".join(lua_parts)
        if s.startswith("sizeof"):
            s = "ffi.sizeof('" + stripSizeOf(s) + "')"
        return s
    elif k == TyK.FLOAT or k == TyK.DOUBLE:
        if "FLT_MAX" in s:
            return "math.huge"
        if "FLT_MIN" in s:
            return "-FLT_MIN"
        return s.replace("+", "").replace("f", "")
    elif (k == TyK.POINTER) and (s == "nullptr" or s == "NULL"):
        return "nil"
    elif k == TyK.POINTER and s.startswith('"'):
        return s
    elif k == TyK.LVALUEREFERENCE or k == TyK.RECORD or k == TyK.POINTER:
        # Handle C++ constructor calls like ImVec2(0,0)
        if s.startswith("ImVec2"):
            # FIXED: Generate ffi.new() for struct pointers
            params = s[s.find("(") + 1 : s.find(")")]
            params = params.replace("f", "")
            return f'ffi.new("ImVec2_C", {params})'
        if s.startswith("ImVec4"):
            params = s[s.find("(") + 1 : s.find(")")]
            params = params.replace("f", "")
            return f'ffi.new("ImVec4_C", {params})'
        return "M." + s
    else:
        print(
            f"unknown value type:  {k} {s}  ### parent =  {p.type.spelling}  {p.spelling}"
        )
    return s


# converts a c value to a lua value - used for optional arguments
def luaifyValue(cParent, s):
    return luaifyValueWithType(cParent, s)


def getLuaFunctionOptionalParams(c):
    parameter_opt = None
    token = list(c.get_tokens())
    for p in c.get_arguments():
        for i in range(0, len(token)):
            if (
                token[i].kind == TK.IDENTIFIER
                and token[i].spelling == p.spelling
                and i < len(token) - 2
            ):
                i += 1
                if (
                    token[i].kind == TK.PUNCTUATION
                    and token[i].spelling == "="
                    and i < len(token) - 2
                ):
                    i += 1
                    braceStack = 0
                    start_token_index = i
                    # Walk tokens to find the full default argument expression
                    while i < len(token):
                        current_token = token[i]
                        if current_token.kind == TK.PUNCTUATION:
                            if current_token.spelling in ("(", "{", "["):
                                braceStack += 1
                            elif current_token.spelling in (")", "}", "]"):
                                if braceStack > 0:
                                    braceStack -= 1
                                else:  # End of argument
                                    break
                            elif current_token.spelling == "," and braceStack == 0:
                                break  # End of argument
                        i += 1

                    # Reconstruct the default argument string from tokens
                    optArg = " ".join(t.spelling for t in token[start_token_index:i])

                    # Post-process to fix spacing issues
                    optArg = (
                        optArg.replace(" (", "(")
                        .replace(" )", ")")
                        .replace(" ,", ",")
                        .replace(" | ", "|")
                    )

                    if parameter_opt is None:
                        parameter_opt = {}
                    param_name_lua = luaParameterSpelling(p, True)
                    parameter_opt[param_name_lua] = luaifyValue(p, optArg.strip())
                break
    return parameter_opt


###############################################################################

# conventions:
#  *C VM* = C code for inside the lua VM, as in the FFI defitions
#  *C Host* = Code for the Lua Host, that exports the FFI bindings
#  *Lua VM* = Lua code for inside the VM that does helper things like optional args


class BindingGenerator:
    def __init__(self, debug):
        self.functionRenames = {}
        self.debug = debug

    ## structs
    def _generateCVMStruct(self, c, level):
        functionCache = ""
        res = ""
        if c.kind == CK.STRUCT_DECL:
            if level == 0:
                res += "  " * (level - 1) + "typedef struct " + c.spelling + " {\n"
            else:
                res += "  " * (level - 1) + "struct " + c.spelling + " {\n"
        elif c.kind == CK.UNION_DECL:
            res += "\n" + "  " * (level - 1) + "union {\n"

        for ch in c.get_children():
            if ch.kind == CK.FIELD_DECL:
                if ch.type.spelling.find("(") >= 0:
                    res += (
                        "  " * (level + 1)
                        + "void* "
                        + ch.spelling
                        + "; // complex callback: "
                        + ch.type.spelling
                        + " - "
                        + self.getCursorDebug(ch, "")
                        + "\n"
                    )
                else:
                    res += (
                        "  " * (level + 1)
                        + getCVarStr(ch, False, is_ffi_header=True)
                        + ";"
                        + self.getCursorDebug(ch, "   // ")
                        + "\n"
                    )
            elif ch.kind == CK.CONSTRUCTOR and level == 0:
                functionCache += "// " + self._generateCVMFunction(ch, "imgui_", None)
            elif ch.kind == CK.STRUCT_DECL or ch.kind == CK.UNION_DECL:
                res += "  " * (level + 1) + self.getCursorDebug(ch, " // ") + "\n"
                res += self._generateCVMStruct(ch, level + 1)
            elif (
                (ch.kind == CK.FUNCTION_DECL or ch.kind == CK.CXX_METHOD)
                and ch.spelling.find("operator") == -1
                and level == 0
            ):
                if ch.get_usr() in skip_usrs or ch.spelling in skip_names:
                    pass
                else:
                    functionCache += self._generateCVMFunction(
                        ch,
                        "imgui_" + c.spelling + "_",
                        c.spelling + "* " + c.spelling + "_ctx",
                    )

        if c.kind == CK.STRUCT_DECL:
            if level == 0:
                res += "  " * level + "} " + c.spelling + ";\n"
            else:
                res += "  " * level + "};\n"
        elif c.kind == CK.UNION_DECL:
            res += "  " * level + "};\n"
        res += functionCache
        return res

    ## struct member functions
    def _generateLVMStruct(self, c):
        if debug:
            res = "--=== struct " + c.spelling + " === " + c.get_usr() + "\n"
        else:
            res = "--=== struct " + c.spelling + " ===\n"
        for ch in c.get_children():
            if (
                ch.get_usr() in skip_usrs
                or ch.spelling in skip_names
                or ch.kind == CK.CLASS_TEMPLATE
                or ch.kind == CK.FUNCTION_TEMPLATE
            ):
                continue
            if (
                ch.kind == CK.FUNCTION_DECL or ch.kind == CK.CXX_METHOD
            ) and ch.spelling.find("operator") == -1:
                res += self._generateLuaVMFunction(
                    ch,
                    c.spelling + "_",
                    "imgui_" + c.spelling + "_",
                    c.spelling + "_ctx",
                )
            elif ch.kind == CK.CONSTRUCTOR:
                if ch.spelling in skip_constructors:
                    continue
                else:
                    res += self._generateLuaConstructor(ch)
        res += "--===\n"
        return res

    def _generateCHostStruct(self, c):
        res = ""
        for ch in c.get_children():
            if (
                ch.get_usr() in skip_usrs
                or ch.spelling in skip_names
                or ch.kind == CK.CLASS_TEMPLATE
                or ch.kind == CK.FUNCTION_TEMPLATE
            ):
                continue
            if (
                ch.kind == CK.FUNCTION_DECL or ch.kind == CK.CXX_METHOD
            ) and ch.spelling.find("operator") == -1:
                res += self._generateCHostFunction(
                    ch,
                    "imgui_" + c.spelling + "_",
                    c.spelling + "_ctx->",
                    c.spelling + "_ctx",
                    c.type.spelling,
                )
        return res

    ## LVM Constructors
    def _generateLuaConstructor(self, c):
        (
            signature,
            resStr,
            parameter_names,
            isVariadic,
            parameter_deref,
            parameter_wrappers,
        ) = self.getCFunctionSignature(c, "", None, False)
        i = 0
        for param in parameter_names:
            if param and param[0] == "_":
                parameter_names[i] = param[1:]
                i += 1
        func = "function M." + c.spelling + "(" + ", ".join(parameter_names) + ")"
        funcPtr = "function M." + c.spelling + "Ptr(" + ", ".join(parameter_names) + ")"
        if len(parameter_names) > 0:
            func += '\n  local res = ffi.new("' + c.spelling + '")\n'
            for param in parameter_names:
                func += "  res." + param + " = " + param + "\n"
            func += "  return res\n"
        else:
            func += ' return ffi.new("' + c.spelling + '") '
            funcPtr += ' return ffi.new("' + c.spelling + '[1]") '
        func += "end\n"
        funcPtr += "end\n"
        return func + funcPtr

    ## functions
    def _generateCVMFunction(self, c, prefix, firstArg):
        signature, _, _, _, _, _ = self.getCFunctionSignature(
            c, prefix, firstArg, False, is_ffi_header=True
        )
        return signature + ";" + self.getCursorDebug(c, "   // ") + "\n"

    def _generateCHostFunction(self, c, prefix, cNamespace, firstArgName, firstArgType):
        firstArg = None
        functionAppendix = ""
        if firstArgName and firstArgType:
            firstArg = firstArgType + "* " + firstArgName
        (
            signature,
            resStr,
            parameter_names,
            isVariadic,
            parameter_deref,
            parameter_wrappers,
        ) = self.getCFunctionSignature(c, prefix, firstArg, True)
        res = ""
        if self.debug:
            res += "\n" + self.getCursorDebug(c, "// ") + "\n"
        res += "FFI_EXPORT " + signature + " {\n"
        if isVariadic:
            functionAppendix = "V"
            if "fmt" in parameter_names:
                last_param_name = "fmt"
            else:
                last_param_name = parameter_names[-1] if parameter_names else ""
            parameter_names.append("args")
            parameter_deref.append(False)
            parameter_wrappers.append(("", ""))
            res += "  va_list args;\n"
            res += f"  va_start(args, {last_param_name});\n"

        paramArr = []
        for i in range(0, len(parameter_names)):
            p_name = parameter_names[i]
            if parameter_deref[i]:
                p_name = "*" + p_name

            # Apply wrapper (e.g., for ImTextureRef)
            wrapper = parameter_wrappers[i]
            p_name = wrapper[0] + p_name + wrapper[1]

            paramArr.append(p_name)
        paramStr = ", ".join(paramArr)

        rt = c.result_type
        if c.result_type.kind == TyK.TYPEDEF:
            rt = c.result_type.get_canonical()

        call_str = cNamespace + c.spelling + functionAppendix + "(" + paramStr + ")"

        if rt.spelling == "ImVec2":
            res += f"  const ImVec2& res_cxx = {call_str};\n"
            res += "  ImVec2_C res_c = {res_cxx.x, res_cxx.y};\n"
            res += "  return res_c;\n"
        elif rt.spelling == "ImVec4" or rt.spelling == "ImColor":
            res += f"  const ImVec4& res_cxx = {call_str};\n"
            res += "  ImVec4_C res_c = {res_cxx.x, res_cxx.y, res_cxx.z, res_cxx.w};\n"
            res += "  return res_c;\n"
        else:
            res += "  " + resStr + call_str + ";\n"
        if isVariadic:
            res += "  va_end(args);\n"
        res += "}\n\n"
        return res

    def _generateLuaVMFunction(self, c, prefixLua, prefixC, firstArg):
        (
            signature,
            resStr,
            parameter_names,
            isVariadic,
            parameter_deref,
            _,
        ) = self.getCFunctionSignature(c, "imgui_", None, False)
        parameters = []
        parameter_opt = getLuaFunctionOptionalParams(c)
        parameter_PtrChecks = {}
        for p in c.get_arguments():
            if p.spelling != "ctx":
                param_lua_name = luaParameterSpelling(p, True)
                parameters.append(param_lua_name)
                if p.type.spelling.find("*") != -1:
                    parameter_PtrChecks[param_lua_name] = p.type.spelling

        multiLineFunction = False
        if firstArg:
            parameters.insert(0, firstArg)

        lua_call_params = list(parameters)
        if isVariadic:
            parameters.append("...")
            lua_call_params.append("...")

        res = ""
        if self.debug:
            res += "\n" + self.getCursorDebug(c, "-- ") + "\n"
            multiLineFunction = True
        res += (
            "function M."
            + prefixLua
            + self.getFunctionName(c)
            + "("
            + ", ".join(parameters)
            + ") "
        )
        if parameter_opt:
            multiLineFunction = True
            res += "\n"
            for k, v in parameter_opt.items():
                if v == "nil":
                    res += "  -- " + k + " is optional and can be nil\n"
                else:
                    if v == "-FLT_MIN":
                        v = "M.ImVec2( -FLT_MIN, 0)"
                    res += "  if " + k + " == nil then " + k + " = " + v + " end\n"

        if len(parameter_PtrChecks) > 0:
            if not multiLineFunction:
                res += "\n"
            multiLineFunction = True
            for k, v in parameter_PtrChecks.items():
                if parameter_opt and k in parameter_opt and parameter_opt[k] == "nil":
                    continue
                res += (
                    "  if "
                    + k
                    + ' == nil then log("E", "", "Parameter \''
                    + k
                    + "' of function '"
                    + self.getFunctionName(c)
                    + "' cannot be nil, as the c type is '"
                    + v
                    + "'\") ; return end\n"
                )

        if debug:
            res += "\n"
            parameters2 = []
            for p in parameters:
                if p == "...":
                    p = "{...}"
                parameters2.append('" .. dumps(' + p + ') .. "')
            res += (
                '  print("*** calling FFI: '
                + prefixC
                + self.getFunctionName(c)
                + "("
                + (", ".join(parameters2))
                + ')")\n'
            )

        if multiLineFunction:
            res += "  "

        if c.result_type.spelling != "void":
            res += "return "
        res += (
            "C."
            + prefixC
            + self.getFunctionName(c)
            + "("
            + ", ".join(lua_call_params)
            + ")"
        )
        if multiLineFunction:
            res += "\nend\n"
        else:
            res += " end\n"
        return res

    ## enums
    def _generateCVMEnum(self, c):
        name = c.spelling
        constants = []
        for ch in c.get_children():
            if ch.kind == CK.ENUM_CONSTANT_DECL:
                value = ""
                for ca in ch.get_children():
                    if ca.kind == CK.UNEXPOSED_EXPR or ca.kind == CK.BINARY_OPERATOR:
                        value = " = " + getContent(ca, False)
                        break
                constants.append("  " + ch.spelling + value)
        if len(constants) == 0:
            res = "typedef " + c.enum_type.get_canonical().spelling + " " + name + ";\n"
            return res
        res = self.getCursorDebug(c, "// ") + "\n"
        res = res + "typedef enum {\n" + ",\n".join(constants) + "\n} " + name + ";\n"
        return res

    def _generateLVMEnum(self, c):
        res = "--=== enum " + c.spelling + " ===\n"
        for ch in c.get_children():
            if ch.kind == CK.ENUM_CONSTANT_DECL:
                lname = ch.spelling
                if lname.startswith("ImGui"):
                    lname = lname[5:]
                res += "M." + lname + " = C." + ch.spelling + "\n"
        res += "--===\n"
        return res

    ## main
    def _traverse(self, c, level):
        if c.location.file and not c.location.file.name.endswith(self.sFilename):
            return

        if (
            c.get_usr() in skip_usrs
            or c.spelling in skip_names
            or c.kind == CK.CLASS_TEMPLATE
            or c.kind == CK.FUNCTION_TEMPLATE
        ):
            return

        if c.kind == CK.FUNCTION_DECL or c.kind == CK.CXX_METHOD:
            if c.spelling.find("operator") == 0:
                return
            self.tVMFile.write(self._generateCVMFunction(c, "imgui_", None))
            self.tHostFile.write(
                self._generateCHostFunction(c, "imgui_", "ImGui::", None, None)
            )
            self.tLuaFile.write(self._generateLuaVMFunction(c, "", "imgui_", None))
            return
        elif c.kind == CK.TYPEDEF_DECL:
            txt = getContent(c, False)
            self.tVMFile.write(txt + ";\n")
            return
        elif c.kind == CK.STRUCT_DECL or c.kind == CK.UNION_DECL:
            if c.is_definition():
                self.tVMFile.write(self._generateCVMStruct(c, 0))
                self.tHostFile.write(self._generateCHostStruct(c))
                self.tLuaFile.write(self._generateLVMStruct(c))
            else:
                self.tVMFile.write(
                    "typedef struct " + c.spelling + " " + c.spelling + ";\n"
                )
            return
        elif c.kind == CK.ENUM_DECL:
            self.tVMFile.write(self._generateCVMEnum(c))
            self.tLuaFile.write(self._generateLVMEnum(c))
            return
        elif c.kind == CK.TRANSLATION_UNIT or c.kind == CK.NAMESPACE:
            pass
        else:
            print(
                "* unhandled item: " + " " * level,
                str(c.kind)[str(c.kind).index(".") + 1 :],
                c.type.spelling,
                c.spelling,
            )
            print(" " * level, "  ", getContent(c, True))

        for cn in c.get_children():
            self._traverse(cn, level + 1)

    def generate(self, c, sFilename):
        self.sFilename = sFilename
        outDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "generated")
        if not os.path.exists(outDir):
            os.mkdir(outDir)

        with open(
            os.path.join(outDir, "imgui_gen.lua"), "w", encoding="utf-8"
        ) as self.tLuaFile:
            self.tLuaFile.write(
                """-- !!!! DO NOT EDIT THIS FILE -- It was automatically generated by gen.py -- DO NOT EDIT THIS FILE !!!!

local ffi = require('ffi')
local C -- Will be initialized with the library

local M = {}

-- Define a placeholder for FLT_MIN that we can use in default arguments
local FLT_MIN = -3.402823466e+38

function M.init(lib)
    C = lib

"""
            )
            with open(
                os.path.join(outDir, "imgui_gen.h"), "w", encoding="utf-8"
            ) as self.tVMFile:
                self.tVMFile.write(
                    """///////////////////////////////////////////////////////////////////////////////
// this file is used for declaring C types for LuaJIT's FFI. Do not use it in C
///////////////////////////////////////////////////////////////////////////////

// !!!! DO NOT EDIT THIS FILE -- It was automatically generated by gen.py -- DO NOT EDIT THIS FILE !!!!

typedef struct ImVector {
    int   Size;
    int   Capacity;
    void* Data;
} ImVector;

typedef struct { float x, y; } ImVec2_C;
typedef struct { float x, y, z, w; } ImVec4_C;

// Typedef for ImTextureID for FFI, as ImGui uses ImU64
typedef unsigned long long ImTextureID;

"""
                )
                with open(
                    os.path.join(outDir, "imguiApiHostGenerated.cpp"),
                    "w",
                    encoding="utf-8",
                ) as self.tHostFile:
                    self.tHostFile.write(
                        """// !!!! DO NOT EDIT THIS FILE -- It was automatically generated by gen.py -- DO NOT EDIT THIS FILE !!!!

#if defined(BNG_VERSION)
  #include "imguiApiHost.h"
#else
  #define STANDALONE 1
  #include "imgui.h"
  #include <cstdint>
  #include <cstdarg>

  #if defined(_WIN32)
    #define PLATFORM_WINDOWS
  #endif
#endif // BNG_VERSION

extern "C" {

#if defined(_WIN32)
    #define FFI_EXPORT __declspec(dllexport)
#else
    #define FFI_EXPORT __attribute__((visibility("default")))
#endif

#if defined(STANDALONE)
  typedef struct { float x, y; } ImVec2_C;
  typedef struct { float x, y, z, w; } ImVec4_C;
  typedef ImU64 ImTextureID;
#endif // STANDALONE

"""
                    )
                    self.detectOverloads(c)
                    self._traverse(c, 0)
                    self.tHostFile.write(
                        """

#undef FFI_EXPORT
} // extern C
"""
                    )
            self.tLuaFile.write(
                """
end
return M
"""
            )

    def getCursorDebug(self, c, prefix):
        if not self.debug:
            return ""
        else:
            return prefix + c.get_usr()

    def getFunctionName(self, c):
        u = c.get_usr()
        if u in self.functionRenames:
            return self.functionRenames[u]
        else:
            return c.spelling

    def detectOverloads(self, c):
        fctCache = {}
        self._rec_detectOverloads(fctCache, c, 0, "")
        for k in list(fctCache.keys()):
            if len(fctCache[k]) == 1:
                del fctCache[k]
        for k, v in fctCache.items():
            for i in range(len(v)):
                self.functionRenames[v[i].get_usr()] = v[i].spelling + str(i + 1)

    def _rec_detectOverloads(self, fctCache, c, level, prefix):
        if (
            c.kind == CK.STRUCT_DECL
            or c.kind == CK.TRANSLATION_UNIT
            or c.kind == CK.NAMESPACE
        ):
            prefix += c.spelling + "_"
        elif c.kind == CK.FUNCTION_DECL or c.kind == CK.CXX_METHOD:
            if c.spelling.find("operator") == 0:
                return
            uName = prefix + c.spelling
            if not uName in fctCache:
                fctCache[uName] = []
            usr = c.get_usr()
            contained = False
            for f in fctCache[uName]:
                if f.get_usr() == usr:
                    contained = True
                    break
            if not contained:
                fctCache[uName].append(c)
        for cn in c.get_children():
            self._rec_detectOverloads(fctCache, cn, level, prefix)

    def getCFunctionSignature(self, c, prefix, firstArg, isHost, is_ffi_header=False):
        parameters = []
        parameter_names = []
        parameter_deref = []
        parameter_wrappers = []  # For C++ host call, e.g. ImTextureRef( ... )
        isVariadic = c.type.is_function_variadic()

        for p in c.get_arguments():
            parameters.append(getCVarStr(p, False, is_ffi_header=is_ffi_header))
            dereferenceRequired = (
                p.type.kind == TyK.LVALUEREFERENCE or p.type.spelling.endswith(" &")
            )
            parameter_names.append(luaParameterSpelling(p, False))
            parameter_deref.append(dereferenceRequired)

            # Special handling for ImTextureRef for the C++ host wrapper
            if isHost and p.type.spelling == "ImTextureRef":
                parameter_wrappers.append(("ImTextureRef(", ")"))
            else:
                parameter_wrappers.append(("", ""))

        if isVariadic:
            parameters.append("...")
        if firstArg:
            parameters.insert(0, firstArg)

        resStr = "return "
        resType = c.result_type.spelling
        effectiveReturnType = c.result_type
        if c.result_type.kind == TyK.TYPEDEF:
            effectiveReturnType = c.result_type.get_canonical()

        if isHost:
            if effectiveReturnType.spelling == "ImVec2":
                resType = "ImVec2_C"
            elif (
                effectiveReturnType.spelling == "ImVec4"
                or effectiveReturnType.spelling == "ImColor"
            ):
                resType = "ImVec4_C"

        if resType == "void":
            resStr = ""

        signature = (
            resType
            + " "
            + prefix
            + self.getFunctionName(c)
            + "("
            + ", ".join(parameters)
            + ")"
        )
        return (
            signature,
            resStr,
            parameter_names,
            isVariadic,
            parameter_deref,
            parameter_wrappers,
        )


def main():
    if len(sys.argv) != 2:
        print("Usage: gen.py [input]")
        print("Example: gen.py imgui.h")
        sys.exit(1)

    sFilename = sys.argv[1]

    if not os.path.exists(sFilename):
        print(f"Error: Input file not found at '{sFilename}'")
        sys.exit(1)

    # Use clang to parse
    try:
        if os.name == "nt":
            # On Windows, add the default LLVM path to the system PATH if it's not there
            llvm_path = "C:/Program Files/LLVM/bin"
            if os.path.exists(os.path.join(llvm_path, "libclang.dll")):
                os.environ["PATH"] = llvm_path + os.pathsep + os.environ["PATH"]
                clang.cindex.Config.set_library_file(
                    os.path.join(llvm_path, "libclang.dll")
                )
            else:
                print(
                    "Warning: libclang.dll not found in default path 'C:/Program Files/LLVM/bin'."
                )
                # The system might find it anyway if it's in the PATH
        else:
            # On Linux, try a few common locations for libclang
            libclang_paths = [
                "/usr/lib/x86_64-linux-gnu/libclang-14.so",
                "/usr/lib/x86_64-linux-gnu/libclang-12.so",
                "/usr/lib/llvm-14/lib/libclang.so.1",
                "/usr/lib/libclang.so",
            ]
            found_path = None
            for path in libclang_paths:
                if os.path.exists(path):
                    found_path = path
                    break
            if found_path:
                clang.cindex.Config.set_library_file(found_path)
            else:
                print(
                    "Warning: Could not find libclang.so in common paths. Ensure it is installed and discoverable."
                )
    except clang.cindex.LibclangError as e:
        print(f"Error initializing libclang: {e}")
        print(
            "Please ensure LLVM/Clang is installed correctly and its location is known to the system (e.g., in your PATH)."
        )
        sys.exit(1)

    index = clang.cindex.Index.create()

    # Add common include paths to help clang find standard headers
    args = [
        "-x",
        "c++",
        "-std=c++17",
        "-D__CODE_GENERATOR__",
        "-DIMGUI_DISABLE_OBSOLETE_FUNCTIONS",
    ]
    if os.name != "nt":
        args.extend(["-I/usr/include", "-I/usr/include/x86_64-linux-gnu"])

    translation_unit = index.parse(sFilename, args)

    if not translation_unit:
        print("Failed to parse the translation unit.")
        for diag in translation_unit.diagnostics:
            print(diag)
        sys.exit(1)

    # Check for parsing errors
    has_errors = False
    for diag in translation_unit.diagnostics:
        if diag.severity >= clang.cindex.Diagnostic.Error:
            print(f"Clang Error: {diag.spelling} at {diag.location}")
            has_errors = True
    if has_errors:
        print(
            "Clang reported errors while parsing. The generated files may be incorrect."
        )

    BindingGenerator(debug).generate(translation_unit.cursor, sFilename)

    outDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "generated")
    file_path = os.path.join(outDir, "imgui_gen.h")
    content = ""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    content = re.sub(r"\bImVec2\b", "ImVec2_C", content)
    content = re.sub(r"\bImVec4\b", "ImVec4_C", content)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("SUCCESS!")
    print("Output files are located in the 'generated/' directory.")


if __name__ == "__main__":
    main()
