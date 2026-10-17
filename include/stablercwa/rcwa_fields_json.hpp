// Single-solve RCWA fields protocol. The compact JSON reply preserves rcwa_json's far-field
// result; five NPY files carry six Cartesian complex fields and absolute/incident phase + mask.
// Request: {"solve": <one-point rcwa_json request>, "sampling": {"z": [...], "nx": 128,
// "y": 0}, "output_dir": "new directory", "request_id": "unique token"}.
// Files are built in an owned temporary directory and published together by directory rename.
// An existing output directory is rejected; it can never masquerade as a successful new result.
//
// Two-dimensional (crossed grating): "solve" with "mode": "periodic2d" (see rcwa_json.hpp; exactly one incidence
// point and an "input"), and a sampling of a vertical slice:
//   "sampling": {"z": [...], "plane": "xz" | "yz", "nt": 128, "at": 0.4}
//   plane "xz": y = at fixed, nt points along x (x_j = j*Lx/nt, the period end is not repeated); plane "yz": x = at fixed,
//   along y (y_j = j*Ly/nt). plane and nt are required, at defaults to 0; nx and y of the 1D form are unknown keys here,
//   plane/nt/at are unknown keys in the 1D form. nt*len(z) <= kFieldsJsonMaxSamples; z is non-empty and finite, order and
//   duplicates are kept. The reply head is the far-field head with one result, then "protocol" and "field_data" (the
//   manifest). Files: fields.npy <c16 [6, nz, nt], phase.npy <f8 [2, 6, nz, nt] (0: absolute, 1: relative to the incident
//   carrier), phase_mask.npy |u1 [2, 6, nz, nt] (1 = undefined, phase NaN), t.npy, x.npy, y.npy <f8 [nt] (x.npy and y.npy
//   are the per-point coordinates of the slice: for "xz" x = t and y = at everywhere, for "yz" x = at and y = t), z.npy <f8 [nz].
//   The manifest has "slice": {plane, at, period}, "harmonics": {M, My, N} and no top-level "y"; every layer (zero
//   thickness included, effective_index = original_index) is listed with finite_zero_modes 0 and schur false.
//   A failed solve (Rayleigh anomaly, numerical failure), an unavailable field context or a non-finite sample is an
//   {"ok": false} reply and publishes nothing, exactly as in 1D. Full specification: gui/validation/field_reference_2d/PROTOCOL_2D.md.
#pragma once

#include <array>
#include <chrono>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

#include "npy.hpp"
#include "rcwa_json.hpp"        // defines STABLERCWA_RCWA_JSON_CONICAL and STABLERCWA_RCWA_JSON_2D, used below
#include "rcwa1d_fields.hpp"
#if STABLERCWA_RCWA_JSON_CONICAL
#include "rcwa1d_conical_fields.hpp"
#endif
#include "rcwa_phase.hpp"
#if STABLERCWA_RCWA_JSON_2D
#include "rcwa2d_fields.hpp"
#endif

namespace stablercwa {
namespace rcwa {

inline constexpr int kFieldsJsonVersion = 1;
inline constexpr std::size_t kFieldsJsonMaxSamples = 4000000;

namespace fields_json_detail {
namespace fs = std::filesystem;
namespace jd = json_detail;
using Clock = std::chrono::steady_clock;

inline double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

struct Request {
    jd::Request solve;
    RealVector z;
    int nx = 0;
    double y = 0.0;
    int nt = 0;                          // 二维（solve.periodic2d）的切面采样：沿切面 nt 个点，固定坐标 at
    double at = 0.0;
    SlicePlane plane = SlicePlane::XZ;
    fs::path output_dir;
    std::string id;
};

// 一维的 sampling：z、nx、y
inline void read_sampling_1d(const json::Value& sampling, Request& request) {
    jd::check_keys(sampling, "sampling", {"z", "nx", "y"});
    request.nx = jd::integer(jd::member(sampling, "nx", "sampling"), "sampling.nx");
    if (request.nx < 1) jd::fail("sampling.nx", "必须是正整数");
    const auto& depths = jd::member(sampling, "z", "sampling");
    jd::need_array(depths, "sampling.z");
    if (depths.size() == 0 || depths.size() > kFieldsJsonMaxSamples / static_cast<std::size_t>(request.nx))
        jd::fail("sampling", "采样网格为空或超过 4000000 个点");
    request.z.reserve(depths.size());
    for (std::size_t j = 0; j < depths.size(); ++j)
        request.z.push_back(jd::number(depths.at(j), jd::child("sampling.z", j)));
    if (const auto* y = sampling.find("y")) request.y = jd::number(*y, "sampling.y");
    if (!request.solve.conical && request.y != 0.0) jd::fail("sampling.y", "planar 场采样的 y 必须是 0");
}

#if STABLERCWA_RCWA_JSON_2D
// 二维的 sampling：z、plane（"xz" 或 "yz"）、nt、at（可以不写，取 0）
inline void read_sampling_2d(const json::Value& sampling, Request& request) {
    jd::check_keys(sampling, "sampling", {"z", "plane", "nt", "at"});
    const std::string plane = jd::text(jd::member(sampling, "plane", "sampling"), "sampling.plane");
    if (plane == "xz") request.plane = SlicePlane::XZ;
    else if (plane == "yz") request.plane = SlicePlane::YZ;
    else jd::fail("sampling.plane", "只能是 \"xz\"（固定 y，沿 x）或 \"yz\"（固定 x，沿 y）");
    request.nt = jd::integer(jd::member(sampling, "nt", "sampling"), "sampling.nt");
    if (request.nt < 1) jd::fail("sampling.nt", "必须是正整数");
    const auto& depths = jd::member(sampling, "z", "sampling");
    jd::need_array(depths, "sampling.z");
    if (depths.size() == 0 || depths.size() > kFieldsJsonMaxSamples / static_cast<std::size_t>(request.nt))
        jd::fail("sampling", "采样网格为空或超过 4000000 个点");
    request.z.reserve(depths.size());
    for (std::size_t j = 0; j < depths.size(); ++j) {
        request.z.push_back(jd::number(depths.at(j), jd::child("sampling.z", j)));
        if (!std::isfinite(request.z.back())) jd::fail(jd::child("sampling.z", j), "必须是有限的数");
    }
    if (const auto* at = sampling.find("at")) {
        request.at = jd::number(*at, "sampling.at");
        if (!std::isfinite(request.at)) jd::fail("sampling.at", "必须是有限的数");
    }
}
#endif

inline Request read_request(const json::Value& root) {
    jd::need_object(root, "");
    jd::check_keys(root, "", {"solve", "sampling", "output_dir", "request_id"});
    Request request;
    request.solve = jd::read_request(jd::member(root, "solve", ""));
    if (request.solve.points.size() != 1) jd::fail("solve.points", "fields 协议每次只接受一个入射点");
    if (request.solve.conical && request.solve.input == jd::InputKind::None)
        jd::fail("solve.input", "圆锥场采样必须指定入射 Jones 或偏振角");
    const auto& sampling = jd::member(root, "sampling", "");
    jd::need_object(sampling, "sampling");
#if STABLERCWA_RCWA_JSON_2D
    if (request.solve.periodic2d) read_sampling_2d(sampling, request);
    else
#endif
        read_sampling_1d(sampling, request);
    const std::string path = jd::text(jd::member(root, "output_dir", ""), "output_dir");
    if (path.empty() || path.find('\0') != std::string::npos) jd::fail("output_dir", "输出目录不能为空或含 NUL");
    request.output_dir = fs::absolute(fs::u8path(path)).lexically_normal();
    if (request.output_dir.filename().empty()) jd::fail("output_dir", "必须是本次新建的子目录");
    request.id = jd::text(jd::member(root, "request_id", ""), "request_id");
    if (request.id.empty() || request.id.size() > 128) jd::fail("request_id", "必须是 1 到 128 字节的本次请求标记");
    return request;
}

class DirectoryOutput {
public:
    explicit DirectoryOutput(const fs::path& final_path) : final_(final_path) {
        if (fs::exists(final_)) throw std::invalid_argument("fields 输出目录已存在，拒绝覆盖历史文件");
        if (!fs::is_directory(final_.parent_path())) throw std::invalid_argument("fields 输出目录的父目录不存在");
        std::random_device random;
        for (int attempt = 0; attempt < 20; ++attempt) {
            temporary_ = final_.parent_path() / fs::u8path(final_.filename().u8string() + ".tmp-" +
                          std::to_string(random()) + "-" + std::to_string(random()));
            if (fs::create_directory(temporary_)) { owned_ = true; return; }
        }
        throw std::runtime_error("fields 不能建立临时输出目录");
    }
    ~DirectoryOutput() {
        if (owned_) { std::error_code ignored; fs::remove_all(temporary_, ignored); }
    }
    DirectoryOutput(const DirectoryOutput&) = delete;
    DirectoryOutput& operator=(const DirectoryOutput&) = delete;
    fs::path file(const char* name) const { return temporary_ / name; }
    void publish() {
        if (fs::exists(final_)) throw std::runtime_error("fields 输出目录在发布前已存在");
        fs::rename(temporary_, final_);
        owned_ = false;
    }
private:
    fs::path final_, temporary_;
    bool owned_ = false;
};

inline void write_shape(json::Writer& writer, const std::vector<std::size_t>& shape) {
    writer.begin_array(true);
    for (auto value : shape) writer.value(static_cast<long long>(value));
    writer.end_array();
}

inline void write_file_description(json::Writer& writer, const char* key, const char* name,
                                   const char* dtype, const std::vector<std::size_t>& shape) {
    writer.key(key); writer.begin_object(true);
    writer.key("name"); writer.value(name);
    writer.key("dtype"); writer.value(dtype);
    writer.key("shape"); write_shape(writer, shape);
    writer.end_object();
}

inline void write_manifest(json::Writer& writer, const Request& request, const FieldGridXZ& grid,
                           double solve_seconds, double sample_seconds, double phase_seconds, double write_seconds) {
    const auto nx = grid.x.size(), nz = grid.z.size();
    const auto& metadata = grid.metadata;
    writer.key("field_data"); writer.begin_object();
    writer.key("version"); writer.value(kFieldsJsonVersion);
    writer.key("producer"); writer.value("stablercwa.cpp.rcwa_fields");
    writer.key("request_id"); writer.value(request.id);
    writer.key("directory"); writer.value(request.output_dir.u8string());
    writer.key("layout"); writer.value("C");
    writer.key("shape"); write_shape(writer, {nz, nx});
    writer.key("components"); writer.begin_array(true);
    for (const char* name : {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}) writer.value(name);
    writer.end_array();
    writer.key("y"); writer.value(request.y);
    writer.key("phase"); writer.begin_object(true);
    writer.key("references"); writer.begin_array(true); writer.value("absolute"); writer.value("incident"); writer.end_array();
    writer.key("floor"); writer.value(1e-6);
    writer.key("unwrap"); writer.value("none");
    writer.key("unit"); writer.value("rad");
    writer.key("mask_true"); writer.value("undefined");
    writer.end_object();
    writer.key("normalization"); writer.begin_object(true);
    writer.key("E"); writer.value(metadata.normalization.electric_incident_amplitude);
    writer.key("H"); writer.value(metadata.normalization.magnetic_incident_amplitude);
    writer.key("H_convention"); writer.value("Z0 H_physical");
    writer.end_object();
    writer.key("files"); writer.begin_object();
    write_file_description(writer, "fields", "fields.npy", "<c16", {6, nz, nx});
    write_file_description(writer, "phase", "phase.npy", "<f8", {2, 6, nz, nx});
    write_file_description(writer, "phase_mask", "phase_mask.npy", "|u1", {2, 6, nz, nx});
    write_file_description(writer, "x", "x.npy", "<f8", {nx});
    write_file_description(writer, "z", "z.npy", "<f8", {nz});
    writer.end_object();
    writer.key("layers"); writer.begin_array();
    for (const auto& layer : metadata.layers) {
        writer.begin_object(true);
        writer.key("original_index"); writer.value(static_cast<long long>(layer.original_index));
        writer.key("effective_index"); writer.value(layer.effective_index);
        writer.key("z_top"); writer.value(layer.z_top);
        writer.key("z_bottom"); writer.value(layer.z_bottom);
        writer.key("finite_zero_modes"); writer.value(layer.finite_zero_mode_count);
        writer.key("schur"); writer.value(layer.propagation == FieldPropagation::Schur);
        writer.end_object();
    }
    writer.end_array();
    writer.key("timing"); writer.begin_object(true);
    writer.key("solve"); writer.value(solve_seconds);
    writer.key("sample"); writer.value(sample_seconds);
    writer.key("phase"); writer.value(phase_seconds);
    writer.key("write"); writer.value(write_seconds);
    writer.end_object(); writer.end_object();
}

#if STABLERCWA_RCWA_JSON_2D

inline void write_manifest_2d(json::Writer& writer, const Request& request, const FieldGridSlice& grid,
                              double solve_seconds, double sample_seconds, double phase_seconds, double write_seconds) {
    const auto nt = grid.nt(), nz = grid.nz();
    const auto& metadata = grid.metadata;
    writer.key("field_data"); writer.begin_object();
    writer.key("version"); writer.value(kFieldsJsonVersion);
    writer.key("producer"); writer.value("stablercwa.cpp.rcwa_fields");
    writer.key("request_id"); writer.value(request.id);
    writer.key("directory"); writer.value(request.output_dir.u8string());
    writer.key("layout"); writer.value("C");
    writer.key("shape"); write_shape(writer, {nz, nt});
    writer.key("components"); writer.begin_array(true);
    for (const char* name : {"Ex", "Ey", "Ez", "Hx", "Hy", "Hz"}) writer.value(name);
    writer.end_array();
    writer.key("slice"); writer.begin_object(true);               // period 是沿切面方向的周期：xz 取 Lx，yz 取 Ly
    writer.key("plane"); writer.value(slice_plane_name(grid.plane));
    writer.key("at"); writer.value(grid.slice_at);
    writer.key("period"); writer.value(slice_period(metadata, grid.plane));
    writer.end_object();
    writer.key("harmonics"); writer.begin_object(true);
    writer.key("M"); writer.value(request.solve.M);
    writer.key("My"); writer.value(request.solve.My);
    writer.key("N"); writer.value(static_cast<long long>(metadata.orders.size()));
    writer.end_object();
    writer.key("phase"); writer.begin_object(true);
    writer.key("references"); writer.begin_array(true); writer.value("absolute"); writer.value("incident"); writer.end_array();
    writer.key("floor"); writer.value(1e-6);
    writer.key("unwrap"); writer.value("none");
    writer.key("unit"); writer.value("rad");
    writer.key("mask_true"); writer.value("undefined");
    writer.end_object();
    writer.key("normalization"); writer.begin_object(true);
    writer.key("E"); writer.value(metadata.normalization.electric_incident_amplitude);
    writer.key("H"); writer.value(metadata.normalization.magnetic_incident_amplitude);
    writer.key("H_convention"); writer.value("Z0 H_physical");
    writer.end_object();
    writer.key("files"); writer.begin_object();
    write_file_description(writer, "fields", "fields.npy", "<c16", {6, nz, nt});
    write_file_description(writer, "phase", "phase.npy", "<f8", {2, 6, nz, nt});
    write_file_description(writer, "phase_mask", "phase_mask.npy", "|u1", {2, 6, nz, nt});
    write_file_description(writer, "t", "t.npy", "<f8", {nt});
    write_file_description(writer, "x", "x.npy", "<f8", {nt});
    write_file_description(writer, "y", "y.npy", "<f8", {nt});
    write_file_description(writer, "z", "z.npy", "<f8", {nz});
    writer.end_object();
    writer.key("layers"); writer.begin_array();
    for (const auto& layer : metadata.layers) {
        writer.begin_object(true);
        writer.key("original_index"); writer.value(static_cast<long long>(layer.original_index));
        writer.key("effective_index"); writer.value(layer.effective_index);
        writer.key("z_top"); writer.value(layer.z_top);
        writer.key("z_bottom"); writer.value(layer.z_bottom);
        writer.key("finite_zero_modes"); writer.value(0);          // 二维没有这两种机制：恒为 0 / false
        writer.key("schur"); writer.value(false);
        writer.end_object();
    }
    writer.end_array();
    writer.key("timing"); writer.begin_object(true);
    writer.key("solve"); writer.value(solve_seconds);
    writer.key("sample"); writer.value(sample_seconds);
    writer.key("phase"); writer.value(phase_seconds);
    writer.key("write"); writer.value(write_seconds);
    writer.end_object(); writer.end_object();
}

// 二维：solve2d_with_fields，再用 field_slice 采样切面、phase_data 取相位。出错（求解失败、上下文不可用、采样非有限……）
// 一律抛异常，由 solve_fields_json_reply 写成 ok:false 的应答；临时目录由 DirectoryOutput 的析构清掉，不发布。
inline JsonReply run_2d(const Request& request, int indent) {
    DirectoryOutput output(request.output_dir);
    const auto& solve = request.solve;
    const auto& point = solve.points.front();
    jd::ThreadSetting threads(solve.threads);
    json::Writer writer(indent);
    jd::write_head(writer, solve);
    auto started = Clock::now();
    FieldResult2D result = solve2d_with_fields(solve.stack2d, point.wavelength, point.theta_deg, point.phi_deg, solve.M, solve.My,
                                               solve.jones2d, solve.options2d);
    const double solve_seconds = seconds(started);
    if (result.result.status != Status2D::Ok)       // 与一维一样没有“带 null 的成功应答”；原因照 Result2D::message 原文给出
        throw std::runtime_error(result.result.message.empty() ? std::string("fields 求解失败") : result.result.message);
    result.fields.require_available();
    jd::write_result_2d(writer, point, result.result, jd::harmonic_count_2d(solve));
    started = Clock::now();
    const FieldGridSlice grid = field_slice(result.fields, request.z, request.plane, request.nt, request.at);
    const double sample_seconds = seconds(started);
    for (const auto& component : grid.components)
        if (!component.all_finite()) throw std::runtime_error("fields 采样产生非有限复场，拒绝发布");
    const auto nt = grid.nt(), nz = grid.nz();
    double phase_seconds = 0.0;
    double write_seconds = 0.0;
    started = Clock::now();
    {
        npy::Writer fields(output.file("fields.npy"), "<c16", {6, nz, nt});
        for (const auto& component : grid.components)
            for (std::size_t j = 0; j < nz * nt; ++j) fields.value(component.data()[j]);
        fields.finish();
        npy::Writer phase(output.file("phase.npy"), "<f8", {2, 6, nz, nt});
        npy::Writer mask(output.file("phase_mask.npy"), "|u1", {2, 6, nz, nt});
        write_seconds += seconds(started);
        for (int reference = 0; reference < 2; ++reference) for (int component = 0; component < 6; ++component) {
            PhaseOptions options;
            options.reference = reference == 0 ? PhaseReference::Absolute : PhaseReference::Incident;
            options.phase_floor = 1e-6;
            options.unwrap = PhaseUnwrap::None;
            started = Clock::now();
            const auto data = phase_data(grid, static_cast<FieldComponent>(component), options);
            phase_seconds += seconds(started);
            if (data.phase.size() != nz * nt || data.mask.size() != nz * nt)
                throw std::runtime_error("fields 相位尺寸与网格不符");
            started = Clock::now();
            for (std::size_t j = 0; j < nz * nt; ++j) {
                if (!data.mask[j] && !std::isfinite(data.phase[j]))
                    throw std::runtime_error("fields 未掩膜相位为非有限值，拒绝发布");
                phase.value(data.mask[j] ? (std::numeric_limits<double>::quiet_NaN)() : data.phase[j]);
                mask.byte(data.mask[j] ? 1 : 0);
            }
            write_seconds += seconds(started);
        }
        started = Clock::now();
        phase.finish(); mask.finish();
        npy::Writer t(output.file("t.npy"), "<f8", {nt});
        npy::Writer x(output.file("x.npy"), "<f8", {nt});
        npy::Writer y(output.file("y.npy"), "<f8", {nt});
        npy::Writer z(output.file("z.npy"), "<f8", {nz});
        for (double value : grid.t) t.value(value);
        for (double value : grid.x) x.value(value);
        for (double value : grid.y) y.value(value);
        for (double value : grid.z) z.value(value);
        t.finish(); x.finish(); y.finish(); z.finish();
    }
    write_seconds += seconds(started);
    writer.end_array();
    writer.key("protocol"); writer.value("stablercwa-fields");
    write_manifest_2d(writer, request, grid, solve_seconds, sample_seconds, phase_seconds, write_seconds);
    writer.end_object();
    JsonReply reply{true, writer.take()};
    output.publish();
    return reply;
}

#endif   // STABLERCWA_RCWA_JSON_2D

inline JsonReply run(const Request& request, int indent) {
#if STABLERCWA_RCWA_JSON_2D
    if (request.solve.periodic2d) return run_2d(request, indent);
#endif
    DirectoryOutput output(request.output_dir);
    const auto& solve = request.solve;
    const auto& point = solve.points.front();
    jd::ThreadSetting threads(solve.threads);
    json::Writer writer(indent);
    jd::write_head(writer, solve);
    FieldGridXZ grid;
    auto started = Clock::now();
    double solve_seconds = 0.0;
#if STABLERCWA_RCWA_JSON_CONICAL
    if (solve.conical) {
        auto result = solve_conical_with_fields(solve.stack, point.wavelength, point.theta_deg,
                                               point.phi_deg, solve.M, solve.options);
        solve_seconds = seconds(started);
        result.fields.require_available();
        jd::write_result(writer, solve, point, result.result);
        const auto polarized = solve.input == jd::InputKind::Angles
            ? result.result.for_angles(solve.psi_deg, solve.delta_deg) : result.result.for_input(solve.Es, solve.Ep);
        started = Clock::now();
        grid = field_map(result, request.z, request.nx, request.y, std::array<cplx, 2>{polarized.Es, polarized.Ep});
    } else
#endif   // 不编入圆锥入射时，jd::read_request 已经拒绝了 conical 请求
    {
        auto result = solve_with_fields(solve.stack, point.wavelength, point.theta_deg, solve.M,
                                        solve.polarization, solve.options);
        solve_seconds = seconds(started);
        result.fields.require_available();
        jd::write_result(writer, point, result.result);
        started = Clock::now();
        grid = field_map(result, request.z, request.nx);
    }
    const double sample_seconds = seconds(started);
    for (const auto& component : grid.components)
        if (!component.all_finite()) throw std::runtime_error("fields 采样产生非有限复场，拒绝发布");
    const auto nx = grid.x.size(), nz = grid.z.size();
    double phase_seconds = 0.0;
    double write_seconds = 0.0;
    started = Clock::now();
    {
        npy::Writer fields(output.file("fields.npy"), "<c16", {6, nz, nx});
        for (const auto& component : grid.components)
            for (std::size_t j = 0; j < nz * nx; ++j) fields.value(component.data()[j]);
        fields.finish();
        npy::Writer phase(output.file("phase.npy"), "<f8", {2, 6, nz, nx});
        npy::Writer mask(output.file("phase_mask.npy"), "|u1", {2, 6, nz, nx});
        write_seconds += seconds(started);
        for (int reference = 0; reference < 2; ++reference) for (int component = 0; component < 6; ++component) {
            PhaseOptions options;
            options.reference = reference == 0 ? PhaseReference::Absolute : PhaseReference::Incident;
            options.phase_floor = 1e-6;
            options.unwrap = PhaseUnwrap::None;
            started = Clock::now();
            const auto data = phase_data(grid, static_cast<FieldComponent>(component), options);
            phase_seconds += seconds(started);
            if (data.phase.size() != nz * nx || data.mask.size() != nz * nx)
                throw std::runtime_error("fields 相位尺寸与网格不符");
            started = Clock::now();
            for (std::size_t j = 0; j < nz * nx; ++j) {
                if (!data.mask[j] && !std::isfinite(data.phase[j]))
                    throw std::runtime_error("fields 未掩膜相位为非有限值，拒绝发布");
                phase.value(data.mask[j] ? (std::numeric_limits<double>::quiet_NaN)() : data.phase[j]);
                mask.byte(data.mask[j] ? 1 : 0);
            }
            write_seconds += seconds(started);
        }
        started = Clock::now();
        phase.finish(); mask.finish();
        npy::Writer x(output.file("x.npy"), "<f8", {nx});
        npy::Writer z(output.file("z.npy"), "<f8", {nz});
        for (double value : grid.x) x.value(value);
        for (double value : grid.z) z.value(value);
        x.finish(); z.finish();
    }
    write_seconds += seconds(started);
    writer.end_array();
    writer.key("protocol"); writer.value("stablercwa-fields");
    write_manifest(writer, request, grid, solve_seconds, sample_seconds, phase_seconds, write_seconds);
    writer.end_object();
    JsonReply reply{true, writer.take()};
    output.publish();
    return reply;
}
}  // namespace fields_json_detail

inline JsonReply solve_fields_json_reply(const std::string& request, int indent = -1) {
    try {
        const auto parsed = json::parse(request);
        return fields_json_detail::run(fields_json_detail::read_request(parsed), indent);
    } catch (const json_detail::RequestError& error) {
        return json_detail::error_reply(error.message, error.where, indent);
    } catch (const std::exception& error) {
        return json_detail::error_reply(error.what(), "fields", indent);
    } catch (...) {
        return json_detail::error_reply("fields 未识别的求解或输出错误", "fields", indent);
    }
}

}  // namespace rcwa
}  // namespace stablercwa
