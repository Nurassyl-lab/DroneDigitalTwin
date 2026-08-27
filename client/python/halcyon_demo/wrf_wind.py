"""
WRF NetCDF wind sampling for the offshore demo.

WRF U/V/W conventions:
  umet > 0: air moving toward East
  vmet > 0: air moving toward North
  wa   > 0: air moving Up

Project AirSim local coordinates used by the offshore demo are NED:
  X = North, Y = East, Z = Down

The vector passed to World.set_wind_velocity is therefore:
  North = vmet, East = umet, Down = -wa
"""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from netCDF4 import Dataset


EARTH_RADIUS_M = 6378137.0
REQUIRED_VARIABLES = (
    "XLAT",
    "XLONG",
    "HGT",
    "LANDMASK",
    "umet10",
    "vmet10",
    "umet_p",
    "vmet_p",
    "wa_p",
    "z_p",
)


@dataclass(frozen=True)
class WRFWindSample:
    sim_north_m: float
    sim_east_m: float
    sim_down_m: float
    lat: float
    lon: float
    altitude_agl_m: float
    u_east_mps: float
    v_north_mps: float
    w_up_mps: float
    north_mps: float
    east_mps: float
    down_mps: float
    horizontal_speed_mps: float
    direction_to_deg: float
    hgt_m: float
    landmask: float
    outside_region: bool
    outside_dataset: bool
    w_vertical_note: str


class WRFWindField:
    def __init__(
        self,
        nc_path: Path,
        origin_lat: Optional[float] = None,
        origin_lon: Optional[float] = None,
        *,
        time_index: int = 0,
        region_half_size_m: float = 1000.0,
        altitude_min_agl_m: float = 0.0,
        altitude_max_agl_m: float = 100.0,
    ):
        self.nc_path = Path(nc_path)
        self.time_index = int(time_index)
        self.region_half_size_m = float(region_half_size_m)
        self.altitude_min_agl_m = float(altitude_min_agl_m)
        self.altitude_max_agl_m = float(altitude_max_agl_m)
        self.dataset = Dataset(str(self.nc_path), "r")
        self._closed = False

        missing = [name for name in REQUIRED_VARIABLES if name not in self.dataset.variables]
        if missing:
            self.close()
            raise ValueError(f"WRF file is missing required variables: {', '.join(missing)}")

        self.xlat = self._read_2d("XLAT")
        self.xlong = self._read_2d("XLONG")
        self.hgt = self._read_2d("HGT")
        self.landmask = self._read_2d("LANDMASK")
        self.shape = self.xlat.shape
        if self.shape[0] < 2 or self.shape[1] < 2:
            self.close()
            raise ValueError(f"WRF horizontal grid is too small: {self.shape}")

        self.lat_min = float(np.nanmin(self.xlat))
        self.lat_max = float(np.nanmax(self.xlat))
        self.lon_min = float(np.nanmin(self.xlong))
        self.lon_max = float(np.nanmax(self.xlong))
        if (origin_lat is None) != (origin_lon is None):
            self.close()
            raise ValueError("WRF origin latitude and longitude must be provided together.")

        if origin_lat is None:
            self.origin_i, self.origin_j = self._select_offshore_origin_index()
            self.origin_lat = float(self.xlat[self.origin_i, self.origin_j])
            self.origin_lon = float(self.xlong[self.origin_i, self.origin_j])
            self.origin_source = "auto-selected offshore LANDMASK=0 cell"
        else:
            self.origin_lat = float(origin_lat)
            self.origin_lon = float(origin_lon)
            self.origin_source = "explicit WRF origin"

        if not (self.lat_min <= self.origin_lat <= self.lat_max):
            self.close()
            raise ValueError(
                f"WRF origin latitude {self.origin_lat:.7f} is outside file range "
                f"{self.lat_min:.7f}..{self.lat_max:.7f}"
            )
        if not (self.lon_min <= self.origin_lon <= self.lon_max):
            self.close()
            raise ValueError(
                f"WRF origin longitude {self.origin_lon:.7f} is outside file range "
                f"{self.lon_min:.7f}..{self.lon_max:.7f}"
            )

        self.grid_north_m, self.grid_east_m = self._grid_local_m()
        self.north_axis_m = np.nanmedian(self.grid_north_m, axis=1)
        self.east_axis_m = np.nanmedian(self.grid_east_m, axis=0)
        self.north_axis_ascending = self._axis_is_monotonic(self.north_axis_m)
        self.east_axis_ascending = self._axis_is_monotonic(self.east_axis_m)
        if self.north_axis_ascending is None or self.east_axis_ascending is None:
            self.close()
            raise ValueError("WRF grid axes are not monotonic enough for bilinear sampling.")

        if origin_lat is not None:
            nearest_origin = np.unravel_index(
                np.nanargmin(
                    self.grid_north_m * self.grid_north_m
                    + self.grid_east_m * self.grid_east_m
                ),
                self.shape,
            )
            self.origin_i = int(nearest_origin[0])
            self.origin_j = int(nearest_origin[1])
        self.origin_hgt_m = float(self.hgt[self.origin_i, self.origin_j])
        self.origin_landmask = float(self.landmask[self.origin_i, self.origin_j])
        self.dx_m = self._median_axis_spacing(self.east_axis_m)
        self.dy_m = self._median_axis_spacing(self.north_axis_m)

        self.u10_var = self.dataset.variables["umet10"]
        self.v10_var = self.dataset.variables["vmet10"]
        self.u_p_var = self.dataset.variables["umet_p"]
        self.v_p_var = self.dataset.variables["vmet_p"]
        self.w_p_var = self.dataset.variables["wa_p"]
        self.z_p_var = self.dataset.variables["z_p"]

    def close(self):
        if not getattr(self, "_closed", True):
            self.dataset.close()
            self._closed = True

    def startup_summary_lines(self) -> List[str]:
        land_text = "land" if self.origin_landmask >= 0.5 else "water"
        return [
            f"WRF file: {self.nc_path}",
            f"WRF native resolution: DX={self.dx_m:.1f} m DY={self.dy_m:.1f} m",
            f"WRF origin: lat={self.origin_lat:.7f} lon={self.origin_lon:.7f}",
            f"WRF origin source: {self.origin_source}",
            (
                "Simulation wind region: "
                f"North {-self.region_half_size_m:.0f}..{self.region_half_size_m:.0f} m, "
                f"East {-self.region_half_size_m:.0f}..{self.region_half_size_m:.0f} m, "
                f"Altitude {self.altitude_min_agl_m:.0f}..{self.altitude_max_agl_m:.0f} m AGL"
            ),
            (
                f"WRF origin grid: i={self.origin_i} j={self.origin_j} "
                f"HGT={self.origin_hgt_m:.2f} m LANDMASK={self.origin_landmask:.0f} ({land_text})"
            ),
        ]

    def sample_ned(self, position_ned: Sequence[float]) -> WRFWindSample:
        north_m = float(position_ned[0])
        east_m = float(position_ned[1])
        down_m = float(position_ned[2])
        altitude_agl_m = max(0.0, -down_m)
        return self.sample_local(north_m, east_m, altitude_agl_m, down_m)

    def sample_geographic(
        self,
        lat: float,
        lon: float,
        altitude_agl_m: float,
    ) -> WRFWindSample:
        north_m, east_m = self.geographic_to_local_m(lat, lon)
        return self.sample_local(north_m, east_m, altitude_agl_m, -altitude_agl_m)

    def sample_local(
        self,
        north_m: float,
        east_m: float,
        altitude_agl_m: float,
        down_m: float,
    ) -> WRFWindSample:
        i0, i1, tn, outside_north = self._bracket_axis(
            self.north_axis_m,
            north_m,
            self.north_axis_ascending,
        )
        j0, j1, te, outside_east = self._bracket_axis(
            self.east_axis_m,
            east_m,
            self.east_axis_ascending,
        )
        outside_dataset = outside_north or outside_east
        outside_region = (
            abs(north_m) > self.region_half_size_m
            or abs(east_m) > self.region_half_size_m
            or altitude_agl_m < self.altitude_min_agl_m
            or altitude_agl_m > self.altitude_max_agl_m
        )

        block_slice = (slice(i0, i1 + 1), slice(j0, j1 + 1))
        u10 = self._read_2d_block(self.u10_var, block_slice)
        v10 = self._read_2d_block(self.v10_var, block_slice)
        hgt = self.hgt[block_slice]
        landmask = self.landmask[block_slice]
        u_p = self._read_3d_block(self.u_p_var, block_slice)
        v_p = self._read_3d_block(self.v_p_var, block_slice)
        w_p = self._read_3d_block(self.w_p_var, block_slice)
        z_p = self._read_3d_block(self.z_p_var, block_slice)

        u_corner = np.zeros((2, 2), dtype=np.float64)
        v_corner = np.zeros((2, 2), dtype=np.float64)
        w_corner = np.zeros((2, 2), dtype=np.float64)
        w_notes = []
        for ii in range(2):
            for jj in range(2):
                z_agl = z_p[:, ii, jj] - hgt[ii, jj]
                valid_levels = np.isfinite(z_agl) & (z_agl >= 0.0)
                u_corner[ii, jj] = self._interp_uv_profile(
                    altitude_agl_m,
                    u10[ii, jj],
                    z_agl,
                    u_p[:, ii, jj],
                    valid_levels,
                )
                v_corner[ii, jj] = self._interp_uv_profile(
                    altitude_agl_m,
                    v10[ii, jj],
                    z_agl,
                    v_p[:, ii, jj],
                    valid_levels,
                )
                w_value, w_note = self._interp_w_profile(
                    altitude_agl_m,
                    z_agl,
                    w_p[:, ii, jj],
                    valid_levels,
                )
                w_corner[ii, jj] = w_value
                if w_note:
                    w_notes.append(w_note)

        u_east = self._bilinear(u_corner, tn, te)
        v_north = self._bilinear(v_corner, tn, te)
        w_up = self._bilinear(w_corner, tn, te)
        north_wind = v_north
        east_wind = u_east
        down_wind = -w_up
        speed = math.hypot(north_wind, east_wind)
        # Direction TO: 0 deg = North, 90 deg = East.
        direction_to_deg = math.degrees(math.atan2(east_wind, north_wind)) % 360.0
        lat, lon = self.local_to_geographic(north_m, east_m)

        return WRFWindSample(
            sim_north_m=float(north_m),
            sim_east_m=float(east_m),
            sim_down_m=float(down_m),
            lat=float(lat),
            lon=float(lon),
            altitude_agl_m=float(altitude_agl_m),
            u_east_mps=float(u_east),
            v_north_mps=float(v_north),
            w_up_mps=float(w_up),
            north_mps=float(north_wind),
            east_mps=float(east_wind),
            down_mps=float(down_wind),
            horizontal_speed_mps=float(speed),
            direction_to_deg=float(direction_to_deg),
            hgt_m=float(self._bilinear(hgt, tn, te)),
            landmask=float(self._bilinear(landmask, tn, te)),
            outside_region=bool(outside_region),
            outside_dataset=bool(outside_dataset),
            w_vertical_note=", ".join(sorted(set(w_notes))),
        )

    def local_to_geographic(self, north_m: float, east_m: float) -> Tuple[float, float]:
        lat = self.origin_lat + math.degrees(float(north_m) / EARTH_RADIUS_M)
        lon = self.origin_lon + math.degrees(
            float(east_m) / (EARTH_RADIUS_M * math.cos(math.radians(self.origin_lat)))
        )
        return lat, lon

    def geographic_to_local_m(self, lat: float, lon: float) -> Tuple[float, float]:
        north_m = math.radians(float(lat) - self.origin_lat) * EARTH_RADIUS_M
        east_m = (
            math.radians(float(lon) - self.origin_lon)
            * EARTH_RADIUS_M
            * math.cos(math.radians(self.origin_lat))
        )
        return north_m, east_m

    def _read_2d(self, name: str) -> np.ndarray:
        var = self.dataset.variables[name]
        data = var[self.time_index, :, :] if len(var.dimensions) == 3 else var[:, :]
        return self._filled_array(data)

    def _read_2d_block(self, var, block_slice) -> np.ndarray:
        data = var[self.time_index, block_slice[0], block_slice[1]]
        return self._ensure_2x2(self._filled_array(data))

    def _read_3d_block(self, var, block_slice) -> np.ndarray:
        data = var[self.time_index, :, block_slice[0], block_slice[1]]
        array = self._filled_array(data)
        if array.ndim != 3 or array.shape[1:] != (2, 2):
            raise ValueError(f"Unexpected WRF vertical block shape: {array.shape}")
        return array

    @staticmethod
    def _filled_array(data) -> np.ndarray:
        if np.ma.isMaskedArray(data):
            data = np.ma.filled(data, np.nan)
        return np.asarray(data, dtype=np.float64)

    @staticmethod
    def _ensure_2x2(array: np.ndarray) -> np.ndarray:
        if array.shape != (2, 2):
            raise ValueError(f"Unexpected WRF horizontal block shape: {array.shape}")
        return array

    def _grid_local_m(self) -> Tuple[np.ndarray, np.ndarray]:
        north_m = np.radians(self.xlat - self.origin_lat) * EARTH_RADIUS_M
        east_m = (
            np.radians(self.xlong - self.origin_lon)
            * EARTH_RADIUS_M
            * math.cos(math.radians(self.origin_lat))
        )
        return north_m, east_m

    @staticmethod
    def _axis_is_monotonic(axis: np.ndarray) -> Optional[bool]:
        diffs = np.diff(axis)
        finite = diffs[np.isfinite(diffs)]
        if finite.size == 0:
            return None
        positive = np.count_nonzero(finite > 0.0)
        negative = np.count_nonzero(finite < 0.0)
        if positive >= max(1, negative * 20):
            return True
        if negative >= max(1, positive * 20):
            return False
        return None

    def _select_offshore_origin_index(self) -> Tuple[int, int]:
        offshore_mask = (
            np.isfinite(self.xlat)
            & np.isfinite(self.xlong)
            & np.isfinite(self.hgt)
            & (self.landmask < 0.5)
        )
        candidates = np.argwhere(offshore_mask)
        if candidates.size == 0:
            raise ValueError(
                "No offshore WRF origin could be auto-selected because LANDMASK==0 "
                "cells were not found."
            )

        center = np.array([(self.shape[0] - 1) / 2.0, (self.shape[1] - 1) / 2.0])
        distances = np.sum((candidates.astype(np.float64) - center) ** 2, axis=1)
        selected = candidates[int(np.argmin(distances))]
        return int(selected[0]), int(selected[1])

    @staticmethod
    def _median_axis_spacing(axis: np.ndarray) -> float:
        diffs = np.abs(np.diff(axis))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if diffs.size == 0:
            return 0.0
        return float(np.nanmedian(diffs))

    @staticmethod
    def _bracket_axis(
        axis: np.ndarray,
        value: float,
        ascending: bool,
    ) -> Tuple[int, int, float, bool]:
        work_axis = axis if ascending else axis[::-1]
        outside = bool(value < work_axis[0] or value > work_axis[-1])
        if value <= work_axis[0]:
            low = 0
            t = 0.0
        elif value >= work_axis[-1]:
            low = len(work_axis) - 2
            t = 1.0
        else:
            high = int(np.searchsorted(work_axis, value, side="right"))
            low = max(0, min(high - 1, len(work_axis) - 2))
            denom = work_axis[low + 1] - work_axis[low]
            t = 0.0 if abs(denom) <= 1e-9 else (value - work_axis[low]) / denom
            t = max(0.0, min(1.0, float(t)))

        if ascending:
            return low, low + 1, t, outside

        max_index = len(axis) - 1
        i0 = max_index - (low + 1)
        i1 = max_index - low
        return i0, i1, t, outside

    @staticmethod
    def _bilinear(values: np.ndarray, tn: float, te: float) -> float:
        return float(
            (1.0 - tn) * (1.0 - te) * values[0, 0]
            + (1.0 - tn) * te * values[0, 1]
            + tn * (1.0 - te) * values[1, 0]
            + tn * te * values[1, 1]
        )

    def _interp_uv_profile(
        self,
        altitude_agl_m: float,
        value_10m: float,
        z_agl: np.ndarray,
        values: np.ndarray,
        valid_levels: np.ndarray,
    ) -> float:
        heights = np.concatenate(([10.0], z_agl[valid_levels]))
        profile_values = np.concatenate(([value_10m], values[valid_levels]))
        return self._interp_profile(altitude_agl_m, heights, profile_values)[0]

    def _interp_w_profile(
        self,
        altitude_agl_m: float,
        z_agl: np.ndarray,
        values: np.ndarray,
        valid_levels: np.ndarray,
    ) -> Tuple[float, str]:
        value, clamp_note = self._interp_profile(
            altitude_agl_m,
            z_agl[valid_levels],
            values[valid_levels],
        )
        return value, clamp_note

    @staticmethod
    def _interp_profile(
        altitude_agl_m: float,
        heights: np.ndarray,
        values: np.ndarray,
    ) -> Tuple[float, str]:
        heights = np.asarray(heights, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        mask = np.isfinite(heights) & np.isfinite(values)
        heights = heights[mask]
        values = values[mask]
        if heights.size == 0:
            return 0.0, "no valid vertical levels"

        order = np.argsort(heights)
        heights = heights[order]
        values = values[order]
        unique_heights = []
        unique_values = []
        for height, value in zip(heights, values):
            if unique_heights and abs(height - unique_heights[-1]) <= 1e-6:
                unique_values[-1] = value
            else:
                unique_heights.append(float(height))
                unique_values.append(float(value))

        heights = np.asarray(unique_heights, dtype=np.float64)
        values = np.asarray(unique_values, dtype=np.float64)
        if heights.size == 1:
            return float(values[0]), "nearest vertical level"

        note = ""
        if altitude_agl_m < heights[0]:
            note = "nearest below vertical range"
        elif altitude_agl_m > heights[-1]:
            note = "nearest above vertical range"
        return float(np.interp(altitude_agl_m, heights, values)), note
