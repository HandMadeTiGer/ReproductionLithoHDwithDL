#ifndef __DATA_TYPE_H__
#define __DATA_TYPE_H__
#include <cstdint>
#include <vector>
#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point_xy.hpp>

typedef int32_t coor_type;
typedef int64_t area_type;
namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
typedef bg::model::d2::point_xy<coor_type> _point;
typedef bg::model::box<_point> _box;
typedef bg::model::polygon<_point, false, false> _polygon;

struct Bbox{
public:
    Bbox(const _point& min, const _point& max){
        box.max_corner().x(max.x());
        box.max_corner().y(max.y());
        box.min_corner().x(min.x());
        box.min_corner().y(min.y());
    }
    Bbox(const _point& min, const _point& max, bool isB, int b)
        : is_block(isB), belong_to(b){
        box.max_corner().x(max.x());
        box.max_corner().y(max.y());
        box.min_corner().x(min.x());
        box.min_corner().y(min.y());
    }
    _box box{};
    bool is_block = false;
    int belong_to = 0;
};

class PolygonsInGDS{
public:
    class ShapeInGDS{
    public:
        ShapeInGDS(std::vector<_point>& ps, int l, int d): shape(ps), layer(l), datatype(d){
            ru.x(INT32_MIN);
            ru.y(INT32_MIN);
            lb.x(INT32_MAX);
            lb.y(INT32_MAX);
            for (const auto &it : shape){
                if (it.x() > ru.x())
                    ru.x(it.x());
                if (it.y() > ru.y())
                    ru.y(it.y());
                if (it.x() < lb.x())
                    lb.x(it.x());
                if (it.y() < lb.y())
                    lb.y(it.y());
            }
        }
        std::vector<_point> shape{};
        int layer{};
        int datatype{};
        _point lb{}, ru{};
    };
    std::vector<ShapeInGDS*> shapes{};
};
typedef std::pair<_box, PolygonsInGDS::ShapeInGDS *> _value;
typedef bgi::rtree<_value, boost::geometry::index::quadratic<128>> _rtree;


#endif // __DATA_TYPE_H__