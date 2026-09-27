# ml/decision_engine.py
"""Phase 4: water-filling assignment with caps, hysteresis, priority, logging."""
import heapq
from common import NETWORKS, CAPACITY, N_DEVICES, SWITCH_THRESHOLD, load_penalty


def L_net(forecast_base, net, k):
    """Predicted latency (ms) for the k-th device placed on `net`."""
    return forecast_base[net] + load_penalty(net, k)


def water_fill(forecast_base, n_devices, caps):
    """Greedy: assign each device to the currently-cheapest network."""
    counts = {net: 0 for net in NETWORKS}
    heap = []                                   # (marginal_latency, net)
    for net in NETWORKS:
        if caps[net] >= 1:
            heapq.heappush(heap, (L_net(forecast_base, net, 1), net))

    order = []
    for _ in range(n_devices):
        if not heap:
            raise RuntimeError("Total capacity is less than number of devices")
        _, net = heapq.heappop(heap)
        counts[net] += 1
        order.append(net)
        nxt = counts[net] + 1
        if nxt <= caps[net]:                    # push this network's next slot
            heapq.heappush(heap, (L_net(forecast_base, net, nxt), net))
    return counts, order


def assign_devices(forecast_base, devices, prev_assignment, caps=CAPACITY):
    """
    devices: list of {"id": str, "priority": bool}
    prev_assignment: {device_id: network} from the previous cycle (may be empty)
    Returns (final_assignment, log_rows, counts).
    """
    # latency-sensitive devices claim the best slots first
    devices = sorted(devices, key=lambda d: not d.get("priority", False))
    _, order = water_fill(forecast_base, len(devices), caps)
    proposed = {dev["id"]: net for dev, net in zip(devices, order)}

    final, log = {}, []
    live = {net: 0 for net in NETWORKS}
    for dev in devices:
        did = dev["id"]
        new_net = proposed[did]
        old_net = prev_assignment.get(did)

        if old_net is None:                     # first ever cycle
            chosen, switched, reason = new_net, True, "initial"
        else:
            old_lat = L_net(forecast_base, old_net, live[old_net] + 1)
            new_lat = L_net(forecast_base, new_net, live[new_net] + 1)
            improve = (old_lat - new_lat) / old_lat if old_lat > 0 else 0.0
            # HYSTERESIS: only move if it helps by more than the threshold
            if new_net != old_net and improve > SWITCH_THRESHOLD:
                chosen, switched, reason = new_net, True, f"improve={improve:.0%}"
            else:
                chosen, switched = old_net, False
                reason = "kept" if new_net == old_net else f"below_thresh={improve:.0%}"

        live[chosen] += 1
        final[did] = chosen
        log.append({"device_id": did, "prev": old_net, "new": chosen,
                    "pred_latency_ms": round(L_net(forecast_base, chosen,
                                                   live[chosen]), 2),
                    "switched": switched, "reason": reason})
    return final, log, live


if __name__ == "__main__":
    fc = {"3G": 130.0, "4G": 55.0, "5G": 14.0}
    devices = [{"id": f"dev{i:03d}", "priority": i < 5} for i in range(N_DEVICES)]

    a1, _, c1 = assign_devices(fc, devices, {})           # cold start
    print("cycle 1 counts:", c1)

    fc2 = {"3G": 130.0, "4G": 95.0, "5G": 14.0}            # 4G congests
    a2, log2, c2 = assign_devices(fc2, devices, a1)
    print("cycle 2 counts:", c2,
          "| switches:", sum(l["switched"] for l in log2), "/", N_DEVICES)