# API reference

The supporting Python surfaces — typed errors, the unit discipline,
shared pydantic base schemas, and the on-disk cache. Tool functions and
their request / response models are documented on the
[Tool reference](tool-reference.md) page.

## Errors

::: astrodynamics_mcp.errors
    options:
      members:
        - AstrodynamicsMCPError
        - InvalidInputError
        - UpstreamError
        - DataSourceError
        - CredentialRequiredError

## Units

::: astrodynamics_mcp.units
    options:
      members:
        - ALLOWED_UNITS
        - Quantity
        - QuantityVector
        - quantity
        - quantity_vector
        - find_unit_discipline_violations
        - is_finite_number

## Schemas

::: astrodynamics_mcp.schemas.base
    options:
      members:
        - TimeScale
        - Frame
        - Epoch
        - Body
        - NamedStation
        - NamedStationName
        - ObserverCoordinates
        - Observer
        - TleLines
        - TleOmm
        - Tle
        - StateVector
        - Interval
        - KeplerianElements

## Cache

::: astrodynamics_mcp.cache
    options:
      members:
        - Cache
        - CacheHit
        - DEFAULT_TTLS
        - default_cache
