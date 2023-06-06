import numpy as np

def block_simulation(sunits, cust, runs, adist, sdist):
    blocked_fractions = np.zeros(runs)
    blocked_total = np.zeros(runs)

    # For each run
    for run in range(runs):

        # Initialize
        sunits_list = np.zeros(sunits)
        total_arrivals = 0
        total_service = 0
        total_blocked = 0

        # Simulation
        for _ in range(cust):
            # Customer arrives
            arrival_time = adist()
            total_arrivals += 1

            # Check if there is a free service unit
            free_units = np.where(sunits_list <= 0)[0]
            if free_units.size > 0:
                # Customer gets served immediately
                unit = free_units[0]
                service_time = sdist()
                sunits_list[unit] = service_time
                total_service += service_time
            else:
                # All service units are busy, customer gets blocked
                total_blocked += 1

            # Decrease service times
            sunits_list = np.maximum(sunits_list - arrival_time, 0)

        # Store the fraction of blocked customers from this run
        blocked_fractions[run] = total_blocked / total_arrivals
        blocked_total[run] = total_blocked
        print(f"Run:{run}: total_blocked:{total_blocked} fraction:{blocked_fractions[run]}")

    # Calculate the mean and standard deviation of the blocked fractions
    mean_blocked = np.mean(blocked_fractions)
    std_blocked = np.std(blocked_fractions)

    # Calculate the 95% confidence intervals
    conf_int = 1.96 * std_blocked / np.sqrt(10)
    upper_limit = mean_blocked + conf_int
    lower_limit = mean_blocked - conf_int

    # Print results
    print(f"Mean total: {np.mean(blocked_total).round(3)} Std total: {np.std(blocked_total).round(3)} Max blocked:{np.max(blocked_total)} Mean fraction: {mean_blocked.round(4)}")
    print(f"Interval: {lower_limit.round(4)} to {upper_limit.round(4)}\n")

# Simulation attributes
service_units = 10
customers = 10000
runs = 10

# Different arrival and service distributions
mean_arrival_time = 1 
mean_service_time = 8

arrival_d = {"erl": lambda: np.random.gamma(shape=2, scale=0.5), 
     "exp": lambda: np.random.exponential(mean_arrival_time), 
     "hexp": lambda: np.random.exponential(1/0.8333) if np.random.uniform() <= 0.8 else np.random.exponential(1/5)}
service_d = {
    "exp": lambda: np.random.exponential(mean_service_time),
    "par": lambda: np.random.pareto(a=1.05),
    "uni": lambda: np.random.uniform(6,10),
    "const": lambda: mean_service_time
}

# Part 1
print("Arrival: Exp , Service: Exp")
block_simulation(sunits=service_units, cust=customers, runs=runs, adist=arrival_d["exp"], sdist=service_d["exp"])

# Part 2
print("Arrival: Erl , Service: Exp")
block_simulation(sunits=service_units, cust=customers, runs=runs, adist=arrival_d["erl"], sdist=service_d["exp"])
print("Arrival: Hexp , Service: Exp")
block_simulation(sunits=service_units, cust=customers, runs=runs, adist=arrival_d["hexp"], sdist=service_d["exp"])

# Part 3
print("Arrival: Exp , Service: Par")
block_simulation(sunits=service_units, cust=customers, runs=runs, adist=arrival_d["exp"], sdist=service_d["par"])
print("Arrival: Exp , Service: Const")
block_simulation(sunits=service_units, cust=customers, runs=runs, adist=arrival_d["exp"], sdist=service_d["const"])
print("Arrival: Exp , Service: Uni")
block_simulation(sunits=service_units, cust=customers, runs=runs, adist=arrival_d["exp"], sdist=service_d["uni"])