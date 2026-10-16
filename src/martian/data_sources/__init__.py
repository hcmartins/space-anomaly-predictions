"""Provider architecture for the Martian Chapter 4 real-data pipeline.

Two abstract interfaces (base.py), each with exactly one real concrete
implementation: EMIRSObservationProvider -> real_emirs_provider.py,
MCDClimatologyProvider -> real_mcd_provider.py. Every provider's output
carries full per-record provenance (provenance.py) and is validated against
a config-driven schema (schema.py) before being returned to a caller.
"""
