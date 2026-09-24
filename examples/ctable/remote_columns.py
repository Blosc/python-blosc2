#!/usr/bin/env python3
"""Bind local, fsspec, and Caterva2 arrays as read-only CTable columns."""

from dataclasses import dataclass

import blosc2


@dataclass
class Reading:
    station_id: int = blosc2.field(blosc2.int32())
    temperature: float = blosc2.field(blosc2.float32())
    humidity: float = blosc2.field(blosc2.float32())


table = blosc2.CTable(
    Reading,
    sources={
        "station_id": blosc2.open("s3://weather/station-id.b2nd", lazy=True),
        "temperature": blosc2.RemoteArray("https://data.example.org/temperature.b2nd"),
        "humidity": blosc2.RemoteArray(
            blosc2.URLPath("weather/humidity.b2nd", urlbase="https://caterva.example.org")
        ),
    },
)

print(table[table.temperature > 20][["station_id", "humidity"]][:10])
