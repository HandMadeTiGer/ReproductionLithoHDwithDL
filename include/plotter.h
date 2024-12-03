#ifndef __PLOTTER_H__
#define __PLOTTER_H__


#include "rtree.h"
#include "data_type.h"

#include <cairo/cairo.h>
#include <iostream>
#include <vector>
class Plotter
{
public:
    Plotter(const _point &max, const _point &min) : cario_(nullptr), surface_(nullptr), max_(max), min_(min){};
    int plot_points(const std::vector<_point> &points, const _point &max, const _point &min, std::vector<double> color);

    int plot_shape_in_gds(_polygon shape, _point &max, _point &min){
        std::vector<_point> ps;
        for(const auto& it : shape.outer())
            ps.emplace_back(it.x(), it.y());
        plot_points(ps, max, min, {1.0, 1.0, 1.0});
        return 0;
    }
    int plot_shapes_in_gds(std::vector<_polygon>& shapes, _polygon& rect){
        _point max;
        max.x(shapes[0].outer()[0].x());
        max.y(shapes[0].outer()[0].y());
        _point min;
        min.x(shapes[0].outer()[0].x());
        min.y(shapes[0].outer()[0].y());
        for (const auto &it : rect.outer()){
            if (it.x() > max.x())
                max.x(it.x());
            if (it.y() > max.y())
                max.y(it.y());
            if (it.x() < min.x())
                min.x(it.x());
            if (it.y() < min.y())
                min.y(it.y());
        }
        for (const auto &shape : shapes)
            plot_shape_in_gds(shape, max, min);
        
        return 0;
    }

    void build()
    {
        factor_ = (max_.x() - min_.x()) / 1000 < 1 ? 1 : (max_.x() - min_.x()) / 1000;
        factor_ = (max_.y() - min_.y()) / 1000 < 1 ? 1 : (max_.y() - min_.y()) / 1000;
        // factor_ *= 2.5;
        surface_ = cairo_image_surface_create(CAIRO_FORMAT_A8, (max_.x() - min_.x()) / factor_, (max_.y() - min_.y()) / factor_);
        cario_ = cairo_create(surface_);
        cairo_set_source_rgba(cario_, 0.0, 0.0, 0.0, 0); 
        cairo_paint(cario_);                       
    }
    int save(std::string file_name)
    {
        return cairo_surface_write_to_png(surface_, (file_name + std::string(".png")).c_str());
    }
    unsigned char* matrix(){
        return cairo_image_surface_get_data(surface_);
    }
    int stride(){
        return cairo_image_surface_get_stride(surface_);  
    }

    ~Plotter()
    {
        cairo_destroy(cario_);
        cairo_surface_destroy(surface_);
    }

private:
    cairo_t *cario_{};
    cairo_surface_t *surface_{};
    const _point &max_{};
    const _point &min_{};
    int factor_{};
};


int Plotter::plot_points(const std::vector<_point> &points, const _point &max, const _point &min, std::vector<double> color)
{
    cairo_set_source_rgba(cario_, color[0], color[1], color[2], 1); 
    cairo_set_line_width(cario_, 0);

    cairo_move_to(cario_, double(points[0].x() - min.x()) / double(factor_), double(max.y() - points[0].y()) / double(factor_));

    for (const auto &p : points)
    {
        cairo_line_to(cario_, double(p.x() - min.x()) / double(factor_), double(max.y() - p.y()) / double(factor_));
    }

    cairo_close_path(cario_);
    cairo_stroke_preserve(cario_);
    cairo_fill(cario_);

    return 0;
}

#endif