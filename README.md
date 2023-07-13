# Fleek Network Topology Simulations

## Generating the sample data

The simulation contains a parser to use the data that can be found at https://wonderproxy.com/blog/a-day-in-the-life-of-the-internet/.

```bash
# get the raw ping data
curl https://wp-public.s3.amazonaws.com/pings/pings-2020-07-19-2020-07-20.csv.gz -o pings.csv.gz

# extract it
gzip -d pings.csv.gz

# get the raw server data
curl https://wp-public.s3.amazonaws.com/pings/servers-2020-07-19.csv -o servers.csv

# parse the raw data into a complete latency matrix and metadata csv files
cargo run -r --bin parser -- pings.csv servers.csv
```
