def main() -> None:
    """Entry point for the `sap-cxii-tech-ex-01` CLI command."""
    import uvicorn

    from sap_cxii_tech_ex_01.config import get_settings

    s = get_settings()
    uvicorn.run(
        "sap_cxii_tech_ex_01.api.app:app",
        host=s.api_host,
        port=s.api_port,
        reload=False,
    )
