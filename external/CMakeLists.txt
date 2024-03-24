# Setting this will only affect the folders down from the current one

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#Prevents link conflicts
set(gtest_force_shared_crt on)

# Add the googletest subdirectory
add_subdirectory(googletest)