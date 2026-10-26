
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(NAME test_pt2_basic_conversion
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_pt2_basic_conversion.py $<TARGET_FILE:pnnx>)
set_tests_properties(test_pt2_basic_conversion PROPERTIES LABELS "pt2;pt2_frontend")

add_executable(pnnx_test_exported_program_parameter_roundtrip
    test_exported_program_parameter_roundtrip.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/ir.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/model_stat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/storezip.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/utils.cpp
)
target_include_directories(pnnx_test_exported_program_parameter_roundtrip PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)

add_test(
    NAME test_exported_program_parameter_roundtrip
    COMMAND $<TARGET_FILE:pnnx_test_exported_program_parameter_roundtrip>
)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test_exported_program_parameter_roundtrip")
set_tests_properties(test_exported_program_parameter_roundtrip PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test_exported_program_parameter_roundtrip"
)

set_tests_properties(test_exported_program_parameter_roundtrip PROPERTIES LABELS "pt2;pt2_frontend")
