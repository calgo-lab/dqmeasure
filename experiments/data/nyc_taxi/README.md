# NYC TLC yellow taxi trips, 2024

The scalability experiment's dataset: the twelve monthly files of 2024, 41,169,720 trips and 744 MB, from
the NYC Taxi & Limousine Commission's [trip record data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
The months are served from `https://d37ci6vzurychx.cloudfront.net/trip-data/`, the zone table from
`https://d37ci6vzurychx.cloudfront.net/misc/`. The columns are described in the
[yellow trips data dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf).

Only the zone table is committed; the trips are gitignored and kept out of the docker image. `uv run get-datasets`
downloads all twelve months and the zone table and checks each against the SHA-256 of the file the paper measured.
`prepare-nyc-taxi` then builds the table from the local files. For a quick local run, one month is enough:

```bash
uv run get-datasets
uv run prepare-nyc-taxi --data-dir experiments/data --months 1
```

That prepares January alone, enough for `run-scalability --max-size 10000` and the tests; leaving `--months`
off takes all twelve. It writes `prepared.parquet` and a `manifest.json` with the row counts and the request
count per size.
