#include "main.h"
#include "include/data_type.h"
#include <cstdint>
#include <ctime>
#include <string>
#include <sys/types.h>
#include <vector>
#include <chrono>
#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

using Matrix = torch::Tensor;
std::vector<bool> label;

int stride = 1200;
int width = 1200;
int height = 1200;
int EXT = 0;
int rows = 1200;
int cols = 1200;
double STEP = 2.5;
int AREA = STEP * STEP;

void setParam(double S = 2.5, int r = 1200, int c = 1200, int E = 0){
    EXT = E;
    STEP = S;
    rows = r;
    cols = c; 
    AREA = STEP * STEP;
}


void rasterisation(const unsigned char* matrix, Matrix& m){

    uint8_t* mat = new uint8_t[rows * cols / AREA]();
\
    int newWidth = cols / STEP;
    int newHeight = rows / STEP;


    for (int newY = 0; newY < newHeight; ++newY) {
        for (int newX = 0; newX < newWidth; ++newX) {
            int startX = static_cast<int>(newX * STEP);
            int endX = static_cast<int>((newX + 1) * STEP);
            int startY = static_cast<int>(newY * STEP);
            int endY = static_cast<int>((newY + 1) * STEP);

            endX = std::min(endX, cols);
            endY = std::min(endY, rows);

            int pixelSum = 0;
            int count = 0;
            for (int y = startY; y < endY; ++y) {
                for (int x = startX; x < endX; ++x) {
                    pixelSum += 255 - matrix[y * cols+ x];
                    ++count;
                }
            }

            mat[newY * newWidth + newX] = static_cast<uint8_t>(pixelSum / count);
        }
    }

    m = torch::from_blob(mat, {rows / STEP, cols / STEP}, torch::kUInt8).clone(); 
}

std::vector<bool> extractLabel(){
    return label;
}

void extractFeatThreads_helper(  int thread_idx, int threads_count, char sub_char, bool comp, Rtree* rtree, 
                        std::vector<PolygonsInGDS::ShapeInGDS*>& hotspot_marker_processed, int num, std::vector<int>& idx, std::vector<Matrix>& img){
    for (int i = thread_idx; i < idx.size(); i += threads_count){
        auto* it = hotspot_marker_processed[idx[i]];
        _point lb(it->lb.x() - EXT, it->lb.y() - EXT);
        _point ru(it->ru.x() + EXT, it->ru.y() + EXT);
        _box rect(lb, ru);
        std::vector<PolygonsInGDS::ShapeInGDS*> results;
        rtree->intersect_query(rect, results);
        _polygon hotspot_poly;
        bg::append(hotspot_poly.outer(), _point(it->lb.x() - EXT, it->lb.y() - EXT));
        bg::append(hotspot_poly.outer(), _point(it->ru.x() + EXT, it->lb.y() - EXT));
        bg::append(hotspot_poly.outer(), _point(it->ru.x() + EXT, it->ru.y() + EXT));
        bg::append(hotspot_poly.outer(), _point(it->lb.x() - EXT, it->ru.y() + EXT));       
            
        std::vector<_polygon> all_inter;
        for (auto* res : results){
            _polygon in_box;
            if (sub_char != 't')
                in_box.outer().insert(in_box.outer().end(), res->shape.begin(), res->shape.end());
            else
                in_box.outer().insert(in_box.outer().end(), res->shape.rbegin(), res->shape.rend());            
            std::vector<_polygon> inter;
            bg::intersection(in_box, hotspot_poly, inter);
            all_inter.insert(all_inter.end(), inter.begin(), inter.end());  
   
        }
        
        Plotter plt(ru, lb);
        plt.build();
        plt.plot_shapes_in_gds(all_inter, hotspot_poly);
        auto* mat = plt.matrix();
        rasterisation(mat, img[i]);

    }
}


void extractFeatThreads(int threads_count, char sub_char, bool comp, Rtree* rtree, 
                        std::vector<PolygonsInGDS::ShapeInGDS*>& hotspot_marker_processed,
                        int num, 
                        std::vector<int>& idx,
                        std::vector<Matrix>& img){
    std::vector<std::thread> threads;
    for(int i = 0; i < threads_count - 1; ++i)
        threads.emplace_back(extractFeatThreads_helper, 
                            i, threads_count, sub_char, comp, std::ref(rtree), std::ref(hotspot_marker_processed), num, std::ref(idx), std::ref(img));
    
    extractFeatThreads_helper(threads_count - 1, threads_count, sub_char, comp, std::ref(rtree), std::ref(hotspot_marker_processed), num, std::ref(idx), std::ref(img));
    for(auto& thread : threads)
        thread.join();
}

std::vector<Matrix> extractFeature(int thread_num, const std::string& file_name, char sub_char, bool comp){
    PolygonsInGDS polygons;

    GdsParser::GdsDB::GdsDB db;
    GdsParser::GdsDB::GdsReader gp(db);
    gp(file_name);
    for (const auto& it : db.cells()){
        for (auto& object : it.objects()){
            std::vector<_point> ps;
            if (object.first == GdsParser::GdsRecords::TEXT || object.first == GdsParser::GdsRecords::SREF)
                continue;
            GdsParser::GdsDB::GdsPolygon* ptr = dynamic_cast<GdsParser::GdsDB::GdsPolygon*>(object.second);
            for(int i = 0; i < ptr->size() - 1; ++i){
                auto& p = ptr->coords_[i];
                ps.emplace_back(p.x(), p.y());
            }
	        PolygonsInGDS::ShapeInGDS* shape = new PolygonsInGDS::ShapeInGDS(ps, ptr->layer(), ptr->datatype());
	        polygons.shapes.emplace_back(shape);
        }
    }
    Rtree* rtree;
    std::vector<PolygonsInGDS::ShapeInGDS*> hotspot_marker;    
    PolygonsInGDS polygons_without_marker;
    int counter = 0;
    std::vector<int> hs_idx;
    std::vector<int> nhs_idx;
    for (auto* it : polygons.shapes){
        if (sub_char == '1' && (it->layer == 21 || it->layer == 22))
            continue;
        if(it->layer == 10){
            polygons_without_marker.shapes.emplace_back(it);
        }else if(it->layer == 21 || it->layer == 22 || it->layer == 23){
            hotspot_marker.emplace_back(it);
            if (it->layer == 21 || it->layer == 22){
                hs_idx.push_back(counter);
            }else{
                nhs_idx.push_back(counter);
            }
            counter++;
        }
    }
    rtree = new Rtree(polygons_without_marker);
    int num = hs_idx.size();
    label = std::vector<bool>(hs_idx.size(), true);
    hs_idx.insert(hs_idx.end(), nhs_idx.begin(), nhs_idx.end());
    std::vector<Matrix> img;
    img.resize(hs_idx.size());
    auto label1 = std::vector<bool>(nhs_idx.size(), false);
    label.insert(label.end(), label1.begin(), label1.end());
    extractFeatThreads(thread_num, sub_char, comp, rtree, hotspot_marker, num, hs_idx, img);
    return img;
}

int main(){
    auto start = std::chrono::high_resolution_clock::now();
    extractFeature(8, "/ai/edayhp/handm/feature_extraction/benchmarks/gds/1/test.gds", '1', false);
    extractFeature(8, "/ai/edayhp/handm/feature_extraction/benchmarks/gds/2/test.gds", '2', false);
    extractFeature(8, "/ai/edayhp/handm/feature_extraction/benchmarks/gds/3/test.gds", '3', false);
    extractFeature(8, "/ai/edayhp/handm/feature_extraction/benchmarks/gds/4/test.gds", '4', false);
    extractFeature(8, "/ai/edayhp/handm/feature_extraction/benchmarks/gds/5/test.gds", '5', false);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time: " << duration << " ms" << std::endl;
    return 0;
}




PYBIND11_MODULE(Layout2ImageE2E, m) {
    m.doc() = "Module to export global img variable to NumPy";
    m.def("setParam", &setParam, "Set the parameter needed");
    m.def("extractFeature", &extractFeature, "Convert img variable to tensor");
    m.def("extractLabel", &extractLabel, "Convert label variable to tensor");
}