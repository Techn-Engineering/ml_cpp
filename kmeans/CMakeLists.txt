set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(NAME "Tech Engineering")
set(EMAIL "efekurdoglu@outlook.com")
message("This project has been created by ${NAME}.")
message("Email: ${EMAIL}")
message("Copyright © 2023 Tech Engineering. All rights reserved.")

add_library(library2
            src/kmeans.cpp)
     
target_include_directories(library2 PUBLIC include)
