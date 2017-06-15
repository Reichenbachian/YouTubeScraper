 /* MovementDetect.i */
 %include <std_string.i>
 %include <opencv.i>
 %cv_instantiate_all_defaults
 %module MovementDetect
 %{
 extern int checkFile(std::string path);
 extern bool is_video_file_good(std::string path);
 %}
 
 extern int checkFile(std::string path);
 extern bool is_video_file_good(std::string path);