local env = std.extVar("__ksonnet/environments");
local params = std.extVar("__ksonnet/params").components["basic-auth"];

local basicauth = import "kubeflow/common/basic-auth.libsonnet";
local instance = basicauth.new(env, params);
instance.list(instance.all)
