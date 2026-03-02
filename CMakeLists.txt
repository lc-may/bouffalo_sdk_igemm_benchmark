cmake_minimum_required(VERSION 3.15)

find_package(bouffalo_sdk REQUIRED HINTS $ENV{BL_SDK_BASE})

sdk_set_main_file(main.c)

# Enable maximum optimization for GEMM benchmark source only.
# The SDK sets arch/tune flags (-march=rv32imafcpzpsfoperand_xtheade -mtune=e907) globally;
# these flags add the code-generation level on top.
set_source_files_properties(main.c PROPERTIES
    COMPILE_OPTIONS "-O3;-funroll-loops;-fomit-frame-pointer"
)


project(mac_int8)

