# Experiments

The code behind the paper's three experiments:

- `data-quality-measurement` corrupts HOSP with six error scenarios at five error rates and records every
  DQM's score.
- `downstream-performance` corrupts SynTabFall and records the DQM scores and what the corruption costs a
  classifier.
- `scalability` times every DQM on growing parts of the 2024 NYC taxi trips.

The first two split a dataset into five folds, corrupt the test fold and let every DQM `fit` on the clean
train fold and `score` the corrupted test fold. A run is one `(scenario, error_rate, seed)`. It writes
`results.parquet` to `RESULTS_DIR/<experiment>/<dataset>/<scenario>/<error_rate>/<seed>/`, and a sweep
skips runs that are already there.

## Local

From the repository root:

```bash
uv sync --all-packages
uv run get-datasets
uv run run-sweep
```

That runs data-quality-measurement. `--experiment downstream-performance` runs the other one, and
`--dry-run` lists the runs. The published sweep also runs the LLM-based `SemanticDataAccuracy`, pinned in
`LLMConfig` in `measures.py`. Put `OPENAI_API_KEY=...` in `.env` at the repository root, then:

```bash
uv run --env-file .env run-sweep --llm
```

## k8s

These steps fit the BHT Datexis cluster; adapt the names to yours. Once:

```bash
kubectl apply -f experiments/k8s/pvc-results.yaml
kubectl -n YOUR_NAMESPACE create secret docker-registry datexis-registry-auth --docker-server=registry.datexis.com --docker-username=YOUR_USERNAME --docker-password=YOUR_TOKEN
```

Build and push the image, and set its tag as `image.tag` in [`k8s/values.yaml`](k8s/values.yaml) and in
[`k8s/prepare-data-job.yaml`](k8s/prepare-data-job.yaml):

```bash
docker build --platform linux/amd64 -f experiments/Dockerfile -t registry.datexis.com/YOUR_NAMESPACE/dqmeasure-experiments:TAG .
docker push registry.datexis.com/YOUR_NAMESPACE/dqmeasure-experiments:TAG
```

To launch an experiment, set `experiment:` in `values.yaml`, then:

```bash
uv run python experiments/k8s/generate_values_yaml.py
helm upgrade --install data-quality-measurement experiments/k8s
```

## Scalability

Locally, on one month of taxi trips:

```bash
uv run get-datasets
uv run prepare-nyc-taxi --data-dir experiments/data --months 1
uv run run-scalability --sweep rows --max-size 10000
```

On the cluster, fill the data volume once, then set `experiment: scalability` in `values.yaml` and install:

```bash
kubectl apply -f experiments/k8s/pvc-data.yaml
kubectl apply -f experiments/k8s/prepare-data-job.yaml
kubectl -n YOUR_NAMESPACE wait --for=condition=complete job/prepare-nyc-taxi --timeout=3600s
uv run python experiments/k8s/generate_values_yaml.py
helm upgrade --install scalability experiments/k8s
```

## Results

`measurements/` holds the published results; the notebooks read them and write to `results/`. To copy new
results off the cluster:

```bash
kubectl -n YOUR_NAMESPACE run results-reader --image=busybox --restart=Never --overrides='{"spec":{"volumes":[{"name":"results","persistentVolumeClaim":{"claimName":"dqmeasure-experiments-results"}}],"containers":[{"name":"results-reader","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"name":"results","mountPath":"/results"}]}]}}'
kubectl -n YOUR_NAMESPACE cp results-reader:/results experiments/results
kubectl -n YOUR_NAMESPACE delete pod results-reader
```

To open the notebooks, from the repository root:

```bash
uv run jupyter lab experiments/notebooks
```
