local env = std.extVar("__ksonnet/environments");
local params = std.extVar("__ksonnet/params").components["basic-auth-ingress"];

local basicauth = import "kubeflow/gcp/basic-auth-ingress.libsonnet";
local instance = basicauth.new(env, params);
instance.list(instance.all)
