if(NOT DEFINED SHADER_SOURCE OR NOT DEFINED SHADER_HEADER OR NOT DEFINED SHADER_SYMBOL)
    message(FATAL_ERROR "SHADER_SOURCE, SHADER_HEADER, and SHADER_SYMBOL are required")
endif()

file(READ "${SHADER_SOURCE}" shader_source)
string(FIND "${shader_source}" "#version 450" version_pos)
if(version_pos EQUAL -1)
    message(FATAL_ERROR "No #version 450 directive in ${SHADER_SOURCE}")
endif()

string(REPLACE "#version 450"
    "#version 450\n#undef roundBFloat16\n#define roundBFloat16(v) uintBitsToFloat((floatBitsToUint(v) + 0x7fffu + ((floatBitsToUint(v) >> 16) & 1u)) & 0xffff0000u)"
    shader_source "${shader_source}")

# Match ncnn's embedded-source preprocessing. The RNE override is compiled
# into the application; no shader source file is needed at runtime.
string(FIND "${shader_source}" "#version" version_start)
string(SUBSTRING "${shader_source}" ${version_start} -1 shader_source)
string(REGEX REPLACE "\n +" "\n" shader_source "${shader_source}")
string(REGEX REPLACE "//[^\n]*" "" shader_source "${shader_source}")
string(REGEX REPLACE "[ \t]+" " " shader_source "${shader_source}")
string(REGEX REPLACE "\n[\n]+" "\n" shader_source "${shader_source}")

get_filename_component(header_dir "${SHADER_HEADER}" DIRECTORY)
file(MAKE_DIRECTORY "${header_dir}")
set(shader_text "${header_dir}/${SHADER_SYMBOL}.txt")
file(WRITE "${shader_text}" "${shader_source}")
file(READ "${shader_text}" shader_hex HEX)
string(REGEX REPLACE "([0-9a-fA-F][0-9a-fA-F])" "0x\\1," shader_hex "${shader_hex}")
string(FIND "${shader_hex}" "," tail_comma REVERSE)
if(tail_comma EQUAL -1)
    message(FATAL_ERROR "Failed to encode shader ${SHADER_SOURCE}")
endif()
string(SUBSTRING "${shader_hex}" 0 ${tail_comma} shader_hex)
file(WRITE "${SHADER_HEADER}" "static const char ${SHADER_SYMBOL}[] = {${shader_hex}};\n")
