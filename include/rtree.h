#ifndef __RTREE_H__
#define __RTREE_H__

#include <utility>
#include "data_type.h"

class Rtree
{
public:
    Rtree(PolygonsInGDS &polys){
        auto& shapes = polys.shapes;
        std::vector<_value> box_shape_vec;
        box_shape_vec.reserve(shapes.size());
        for (auto* shape : shapes){
            _box b(shape->lb, shape->ru);
            box_shape_vec.emplace_back(std::make_pair(b, shape));
        }
        tree_ = new _rtree(box_shape_vec.begin(), box_shape_vec.end());
    }
    
    void build(PolygonsInGDS &polygons);
    void insert(PolygonsInGDS::ShapeInGDS *shape);
    size_t intersect_query(_box &bbox, std::vector<PolygonsInGDS::ShapeInGDS *> &result_list);
    size_t intersects_by_query(const _box &bbox, std::vector<PolygonsInGDS::ShapeInGDS *> &queried_shape);
    ~Rtree(){
        delete tree_;
    }
private:
    _rtree* tree_;
};
#endif // __RTREE_H__
