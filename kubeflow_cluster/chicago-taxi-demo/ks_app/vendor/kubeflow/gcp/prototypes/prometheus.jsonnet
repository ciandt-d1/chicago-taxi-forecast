// @apiVersion 0.1
// @name io.ksonnet.pkg.prometheus
// @description Provides prometheus prototype in kubeflow gcp package
// @shortDescription Prometheus Service.
// @param name string Name for the component
// @param projectId string GCP project id.
// @param clusterName string GKE cluster name.
// @param zone string GKE cluster zone.

local prometheus = import "kubeflow/gcp/prometheus.libsonnet";
local instance = prometheus.new(env, params);
instance.list(instance.all)
