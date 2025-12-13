# Locust Load Testing

[Locust](https://docs.locust.io/en/stable/what-is-locust.html) is an open source performance/load testing tool for HTTP and other protocols. Its developer-friendly approach lets you define your tests in regular Python code. Locust tests can be run from command line or using its web-based UI. Throughput, response times and errors can be viewed in real time and/or exported for later analysis.

## Steps

1. Update `host` in `locust_rfdetr.conf` with endpoint name (this will be test run name)
2. Update `users` in `locust_rfdetr.conf` with desired number of concurrent users (1, 25, 50, 100, etc.)
3. Update `endpoint_name` in `locust_rfdetr.py` with endpoint name
4. Update `region` in `locust_rfdetr.py` if different than `us-east-1`
5. Install Python dependencies (see below)
6. Run locust command: `locust --config locust_rfdetr.conf` (if you want run headless without the UI add `--headless`)
7. Optional UI: Go to locust web console: `http://0.0.0.0:8089/`
8. Optional UI: Start the load test

## Install Requirements (Mac with `pip`)

```bash
python -m pip install virtualenv --break-system-packages -Uq
python -m venv .venv
source .venv/bin/activate

python -m pip install pip -Uq
python -m pip install -r requirements.txt -Uq
```
