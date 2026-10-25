# Restores the 2026-10-25 post-06:54Z baseline; never re-enables ECN/WRED.
# Original source: /interface ethernet switch qos export verbose (RouterOS 7.24.5).
# Stop serving first; replay all exported QoS fields, including unset queue rates.
# Captured baseline: select menu.
/interface ethernet switch qos port
# Captured baseline: restore exported values.
set qsfp56-dd-1-1 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-1-2 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-1-3 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-1-4 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-1-5 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-1-6 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-1-7 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-1-8 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-2-1 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-2-2 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-2-3 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-2-4 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-2-5 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-2-6 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-2-7 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-2-8 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-3-1 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-3-2 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-3-3 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-3-4 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-3-5 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-3-6 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-3-7 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-3-8 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-4-1 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-4-2 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-4-3 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-4-4 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-4-5 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-4-6 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-4-7 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set qsfp56-dd-4-8 !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default pfc=disabled profile=default trust-l2=ignore trust-l3=ignore tx-manager=default
# Captured baseline: restore exported values.
set switch1-cpu !egress-rate-queue0 !egress-rate-queue1 !egress-rate-queue2 !egress-rate-queue3 !egress-rate-queue4 !egress-rate-queue5 !egress-rate-queue6 !egress-rate-queue7 map=default profile=default trust-l2=keep trust-l3=keep tx-manager=default
# Captured baseline: select menu.
/interface ethernet switch qos settings
# Captured baseline: restore exported values.
set lossless-buffers=auto lossless-traffic-class=auto mirror-profile=default multicast-buffers=auto shared-buffers=auto wred-threshold=medium
# Captured baseline: select menu.
/interface ethernet switch qos tx-manager queue
# Captured baseline: restore exported values.
set 0 ecn=no schedule=low-priority-group weight=1 wred=no
# Captured baseline: restore exported values.
set 1 ecn=no schedule=low-priority-group weight=2 wred=no
# Captured baseline: restore exported values.
set 2 ecn=no schedule=low-priority-group weight=3 wred=no
# Captured baseline: restore exported values.
set 3 ecn=no schedule=high-priority-group weight=3 wred=no
# Captured baseline: restore exported values.
set 4 ecn=no schedule=high-priority-group weight=4 wred=no
# Captured baseline: restore exported values.
set 5 ecn=no schedule=high-priority-group weight=5 wred=no
# Captured baseline: restore exported values.
set 6 ecn=no schedule=strict-priority wred=no
# Captured baseline: restore exported values.
set 7 ecn=no schedule=strict-priority wred=no
# Remove only the profile names introduced by the lossless import.
/interface ethernet switch qos priority-flow-control remove [find where name="cuteafd-pfc3"]
# Remove the manual RoCE map before deleting its profile.
/interface ethernet switch qos map ip remove [find where profile="cuteafd-roce"]
# Remove the manual CNP map before deleting its profile.
/interface ethernet switch qos map ip remove [find where profile="cuteafd-cnp"]
# Remove only the RoCE profile added by this task.
/interface ethernet switch qos profile remove [find where name="cuteafd-roce"]
# Remove only the CNP profile added by this task.
/interface ethernet switch qos profile remove [find where name="cuteafd-cnp"]
