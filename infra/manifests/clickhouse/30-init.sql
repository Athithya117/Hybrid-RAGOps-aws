CREATE DATABASE IF NOT EXISTS logs;
CREATE TABLE IF NOT EXISTS logs.kube_logs (ts DateTime64(3) DEFAULT now(), service String, pod String, namespace String, message String, fields String, level String, container String, trace_id String, span_id String) ENGINE = MergeTree() ORDER BY ts;
CREATE USER IF NOT EXISTS vector IDENTIFIED WITH plaintext_password BY 'vectorpass';
GRANT INSERT ON logs.* TO vector;
GRANT SELECT ON logs.* TO vector;
