#include "AHCPlaneFitter.hpp"

#include <glob.h> // glob(), globfree()
#include <string.h> // memset()
#include <vector>
#include <stdexcept>
#include <string>
#include <sstream>
#include <fstream>

using namespace std;
std::vector<std::string> globFiles(const std::string& pattern) {
    using namespace std;

    // glob struct resides on the stack
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    // do the glob operation
    int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
    if(return_value != 0) {
        globfree(&glob_result);
        stringstream ss;
        ss << "glob() failed with return_value " << return_value << endl;
        throw std::runtime_error(ss.str());
    }

    // collect all the filenames into a std::list<std::string>
    vector<string> filenames;
    for(size_t i = 0; i < glob_result.gl_pathc; ++i) {
        filenames.push_back(string(glob_result.gl_pathv[i]));
    }

    // cleanup
    globfree(&glob_result);

    // done
    return filenames;
}
std::string get_timestamp(std::string str, int back_cat){
    int len = str.size();
    int time = 0;
    int count = 1;
    for(int i=len - back_cat; i>0; i--){
        if(str[i] == '/') break;
        if(str[i] <= '9' && str[i] >= '0'){
            time += count * (str[i] - '0');
            count *= 10;
        }
    }
    return std::to_string(time);
}
struct OrganizedImage3D {
    const cv::Mat_<cv::Vec3f>& cloud;
    //note: ahc::PlaneFitter assumes mm as unit!!!
    OrganizedImage3D(const cv::Mat_<cv::Vec3f>& c): cloud(c) {}
    inline int width() const { return cloud.cols; }
    inline int height() const { return cloud.rows; }
    inline bool get(const int row, const int col, double& x, double& y, double& z) const {
        const cv::Vec3f& p = cloud.at<cv::Vec3f>(row,col);
        x = p[0];
        y = p[1];
        z = p[2];
        return z > 0 && std::isnan(z)==0; //return false if current depth is NaN
    }
};
typedef ahc::PlaneFitter< OrganizedImage3D > PlaneFitter;
cv::Mat_<cv::Vec3f> getPointCloudSparse(std::string filename,cv::Mat&depth){
    cv::Mat_<cv::Vec3f> cloud(depth.rows, depth.cols);
    std::ifstream infile;
    infile.open(filename);
    
    for(int r=0; r<depth.rows; r++){
        cv::Vec3f* pt_ptr = cloud.ptr<cv::Vec3f>(r);

        for(int c=0; c<depth.cols; c++){
            pt_ptr[c][0] = .0f;
            pt_ptr[c][1] = .0f;
            pt_ptr[c][2] =100000.0f;
        }
    }
    if(!infile.is_open()){std::cout<<"cannot open"<<endl; return cloud;}
        std::string line;
        std::vector<float> tmp(5, .0f);

        while(getline(infile, line)){

            line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());

            std::istringstream ss(line);

            int i=0;
            
            do{
                if(i > 4) break;
                std::string num;
                ss>>num;
                tmp[i] = std::stof(num);
                i++;
            }while(ss);
            auto value = cv::Vec3f(tmp[2], tmp[3], tmp[4]);

            // auto value = cv::Vec3f(tmp[2]*1000.0, tmp[3]*1000.0, tmp[4]*1000.0);
             int x = int(tmp[1]);//auto y = int(tmp[0]);
            float debug_y01 = (720.0f-tmp[0]) / 240.0f;
            int y = int ((1.0 - (debug_y01 * 0.5 +0.5)) * 480);
            if(y<0 || y >=480 || x <0 || x>=640) continue;
            cloud(y,x) = value;
            auto d = std::sqrt(tmp[2] * tmp[2] + tmp[3]*tmp[3] + tmp[4]*tmp[4]);
            depth.at<float>(y,x) = d;
        }
        infile.close();
    
    return cloud;

}
void generate_sparse(std::string point_file, std::string postfix){
    std::cout<<point_file<<std::endl;
    cv::Mat depth = cv::Mat::zeros(480,640,CV_32FC1);
    auto cloud = getPointCloudSparse(point_file, depth);
    auto timestamp= get_timestamp(point_file, 2);
    auto filename = "../sparse_point/"+postfix+"/"+ timestamp + "_sparse.tiff";
    cv::imwrite(filename, depth);
}
void run_pf(std::string dense_file){
    auto timestamp = get_timestamp(dense_file, 10);
    // cv::Mat depth = cv::imread("../resource/1305031103.027881.png",cv::IMREAD_ANYDEPTH);
    // cv::Mat depth = cv::imread("../resource/d32.png",cv::IMREAD_ANYDEPTH);
// cv::Mat depth = cv::imread("../resource/test.png",cv::IMREAD_GRAYSCALE);
cv::Mat depth = cv::imread(dense_file, cv::IMREAD_GRAYSCALE);

//origin
    // const float f = 525;
    // const float cx = 319.5;
    // const float cy = 239.5;
    //nyud2
    // float f = 666;
    // float cx = 325.58;
    // float cy = 253.73;
    //mine
        float f = 667;
    float cx = 316.63575876;
    float cy = 229.10584522;
    const float max_use_range = 10;

    float value = 5000.0f;//255/10
double minVal; 
double maxVal; 
cv::Point minLoc; 
cv::Point maxLoc;

// cv::minMaxLoc( depth, &minVal, &maxVal, &minLoc, &maxLoc );
// value = maxVal / max_use_range;
// std::cout<<"value: "<<value;

    cv::Mat_<cv::Vec3f> cloud(depth.rows, depth.cols);
    for(int r=0; r<depth.rows; r++)
    {
        // const unsigned short* depth_ptr = depth.ptr<unsigned short>(r);
        auto depth_ptr = depth.ptr(r);
        cv::Vec3f* pt_ptr = cloud.ptr<cv::Vec3f>(r);
        for(int c=0; c<depth.cols; c++){
            float z = (float)depth_ptr[c]/value;
            // std::cout<<z<<" ";
            if(z>max_use_range){z=0;}
            pt_ptr[c][0] = (c-cx)/f*z*1000.0;//m->mm
            pt_ptr[c][1] = (r-cy)/f*z*1000.0;//m->mm
            pt_ptr[c][2] = z*1000.0;//m->mm
        }
    }

    /*cv::Mat depth = cv::Mat::zeros(480,640,CV_8UC3);
    auto cloud = getPointCloudSparse("../points/172.txt",depth);*/

    PlaneFitter pf;
    // pf.minSupport = 3000;
    pf.windowWidth = 30;
    pf.windowHeight = 30;
    pf.doRefine = true;

    cv::Mat seg(depth.rows, depth.cols, CV_8UC3);
    OrganizedImage3D Ixyz(cloud);
    std::vector<std::vector<int>> member;
    pf.run(&Ixyz, &member, &seg);
    std::cout<<member.size()<<std::endl;

    // for(auto mem:member){
    //     std::cout<<mem.size()<<std::endl;

    // }


    cv::Mat depth_color;
    depth.convertTo(depth_color, CV_8UC1, 50.0/5000);
    applyColorMap(depth_color, depth_color, cv::COLORMAP_JET);
    // cv::imshow("seg",seg);
    // cv::imshow("depth",depth);
    // cv::waitKey();
    cv::imwrite("../plane_seg/"+  timestamp + ".png", seg);
    
}
int main(int argc, char** argv){
    if(argc > 1){
        std::string func = std::string(argv[1]);
        std::string postrix = std::string(argv[2]);
        std::cout <<func<<" of " << postrix<<std::endl;

        if(func == "gen"){
            generate_sparse(std::string(argv[3]), postrix);
        }else if(func == "genall"){
            auto point_files = globFiles("/home/eevee/Github/mediapipe/mappoints/" + postrix + "/*.txt");
            // auto point_files = globFiles("/home/menghe/Github/mediapipe/mappoints/" + postrix + "/*.txt");
            for(auto pf : point_files) generate_sparse(pf, postrix);
        }else if(func == "run"){
            run_pf(std::string(argv[3]));
        }else if(func == "runall"){
            auto dense_files = globFiles("/home/eevee/Github/sparse-to-dense/res/" + postrix + "/*_dense.png");
            for(auto df : dense_files) run_pf(df);
        }
    } 
  
    return 0; 
} 