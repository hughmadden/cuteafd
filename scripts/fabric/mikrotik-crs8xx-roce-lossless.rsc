# TJ applies only with serving stopped; RouterOS 7.24.5 / CRS804-4DDQ.
# Baseline 2026-10-25 after TJ disabled ECN/WRED on all eight queues.
# PFC-only: traffic byte 104 is Not-ECT; ECN cannot help it. Never enable WRED.
# QoS offload already enabled; v7.23+ sizes lossless buffers automatically.
# Manual IP-only maps avoid promoting unrelated PCP-tagged TCP into TC3/TC6.
# Run once against the captured baseline; duplicate names intentionally fail on repeat.
# Classify only DSCP 26 as RoCE, priority/traffic class 3.
/interface ethernet switch qos profile add name=cuteafd-roce dscp=26 pcp=3 traffic-class=3 automap=no
# Install only the DSCP classifier; default map/profile still catches everything else.
/interface ethernet switch qos map ip add dscp=26 profile=cuteafd-roce
# Classify DSCP 48 CNP into the existing strict-priority class 6.
/interface ethernet switch qos profile add name=cuteafd-cnp dscp=48 pcp=6 traffic-class=6 automap=no
# Map only CNP DSCP 48 into strict-priority TC6; no ECN/DCQCN change is implied.
/interface ethernet switch qos map ip add dscp=48 profile=cuteafd-cnp
# Make traffic class 3 lossless in both directions.
/interface ethernet switch qos priority-flow-control add name=cuteafd-pfc3 traffic-class=3 rx=yes tx=yes
# moa: explicit queue3 wire rate is required to convert received PFC pause time.
/interface ethernet switch qos port set qsfp56-dd-1-1 pfc=cuteafd-pfc3 trust-l2=keep trust-l3=keep egress-rate-queue3=100.0Gbps
# rhea: explicit queue3 wire rate is required to convert received PFC pause time.
/interface ethernet switch qos port set qsfp56-dd-1-5 pfc=cuteafd-pfc3 trust-l2=keep trust-l3=keep egress-rate-queue3=100.0Gbps
# raptor: explicit queue3 wire rate is required to convert received PFC pause time.
/interface ethernet switch qos port set qsfp56-dd-2-1 pfc=cuteafd-pfc3 trust-l2=keep trust-l3=keep egress-rate-queue3=400.0Gbps
# ostrich: explicit queue3 wire rate is required to convert received PFC pause time.
/interface ethernet switch qos port set qsfp56-dd-3-1 pfc=cuteafd-pfc3 trust-l2=keep trust-l3=keep egress-rate-queue3=100.0Gbps
# emu: explicit queue3 wire rate is required to convert received PFC pause time.
/interface ethernet switch qos port set qsfp56-dd-3-5 pfc=cuteafd-pfc3 trust-l2=keep trust-l3=keep egress-rate-queue3=100.0Gbps
# kiwi: explicit queue3 wire rate is required to convert received PFC pause time.
/interface ethernet switch qos port set qsfp56-dd-4-1 pfc=cuteafd-pfc3 trust-l2=keep trust-l3=keep egress-rate-queue3=100.0Gbps
# dodo: explicit queue3 wire rate is required to convert received PFC pause time.
/interface ethernet switch qos port set qsfp56-dd-4-5 pfc=cuteafd-pfc3 trust-l2=keep trust-l3=keep egress-rate-queue3=100.0Gbps
# All other queues/profiles/ports, including default lossy TC1, stay unchanged.
# Do not set use-shared-buffers: CRS8xx v7.23+ manages buffers automatically.
