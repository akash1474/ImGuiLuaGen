# ImGui LuaJIT FFI wrapper Generator

## Setup

This is a python3 script that generates LuaJIT FFI bindings and lua wrappers by parsing the imgui header file via clang.

1. install all the dependencies:

   - on Ubuntu 20.04:
     ```bash
     apt install python3-pip clang libclang-dev libclang-6.0-dev
     pip3 install clang
     ```
   - on Windows 10:
     ensure python3 and pip are installed and usable from command line
     ```bash
     pip3 install clang
     ```
     install LLVM to the default install path, using a prebuilt \*win64.exe installer from: https://github.com/llvm/llvm-project/releases

2. run it like this:

   ```bash
   python gen.py imgui.h
   ```

## Example

```lua
local fColor = Float4(0.9, 0.5, 0.2, 1.0)
local vColor = ImVec4(0.9, 0.5, 0.2, 1.0)
local button_pos = ImVec2(100, 150)
local slider_value = IntPtr(50)
local text_buffer, buffer_len = CString("You can edit this!", 128)


function OnImGuiDraw()
    ImGui.Begin("Advanced Helpers")

    ImGui.Text("Edit the color:")
    if ImGui.ColorEdit4("Color Picker", fColor) then
        print(fColor[0]);
    end

    ImGui.PushStyleColor2(ImGui.Col_Button, vColor:cdata())
    ImGui.Button("I change color!")
    ImGui.PopStyleColor()

    ImGui.Separator()

    local second_button_pos = button_pos + ImVec2(0, 30) -- Move 30 pixels down

    ImGui.Text("Using vector math for layout:")
    ImGui.SetCursorPos(button_pos:cdata())
    if ImGui.Button("Button at Pos A") then
        print("Position A:", button_pos)
    end

    ImGui.SetCursorPos(second_button_pos:cdata())
    if ImGui.Button("Button at Pos B") then
        print("Position B:", second_button_pos)
    end

    ImGui.Text("Drag the slider:")
    if ImGui.SliderInt("Value", slider_value, 0, 100) then
        print("Slider is now:", FromPtr(slider_value))
    end

    ImGui.Separator()

    ImGui.Text("Type in the box below:")
    if ImGui.InputText("Input", text_buffer:cdata(), buffer_len) then
        print("Text was changed to:", text_buffer:get())
    end


    ImGui.ShowDemoWindow();

    ImGui.End()
end
```
