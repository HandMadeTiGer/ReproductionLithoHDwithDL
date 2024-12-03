#include "rtree.h"
void Rtree::build(PolygonsInGDS &layout)
{
    auto& shapes = layout.shapes;
    for (auto *shape : shapes)
        insert(shape);
}
void Rtree::insert(PolygonsInGDS::ShapeInGDS *shape)
{
    _box b(_point(shape->lb.x(), shape->lb.y()), _point(shape->ru.x(), shape->ru.y()));
    tree_->insert(std::make_pair(b, shape));
}
size_t Rtree::intersect_query(_box &bbox, std::vector<PolygonsInGDS::ShapeInGDS *> &queried_shape)
{
    std::vector<_value> result_list;
    size_t t = tree_->query(bgi::intersects(bbox), std::back_inserter(result_list));
    // size_t t = tree_.query(bgi::nearest(b, 1), std::back_inserter(result_list));
    for (const auto &shape : result_list)
        queried_shape.emplace_back(shape.second);
    return t;
}

size_t Rtree::intersects_by_query(const _box &bbox, std::vector<PolygonsInGDS::ShapeInGDS *> &queried_shape)
{
    std::vector<_value> result_list;
    size_t t = tree_->query(bgi::intersects(bbox), std::back_inserter(result_list));
    size_t num = 0;
    for (const auto &shape : result_list){
        _polygon poly;
        for(const auto& p : shape.second->shape){
            bg::append(poly.outer(), _point(p.x(), p.y()));
        }
        if(bg::intersects(bbox, poly)){
            queried_shape.emplace_back(shape.second);
            ++num;
        }
    }
    return num;
}

