import matplotlib.pyplot as plt


def get_data_list(file_path):
    step_list, time_list, acc_list = [], [], []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            try:
                line = line.split()
                step, time, acc = int(line[0]), float(line[1]), float(line[2])
                step_list += [step]
                time_list += [time]
                acc_list += [acc]
            except Exception as e:
                # print(file_path, e)
                pass
    return (step_list, time_list, acc_list)


file_names = (
    "FAISS_IVFPQ_GPU_regular_update.txt",
    "FAISS_IVFPQ_GPU_only_rebuild_no_update_exponential.txt",
    "PyTorchImplementation.txt",
    "FAISS_IVFPQ_CPU_only_rebuild_no_update_exponential.txt"
)


ivfpq_reg_up_data, ivfpq_no_up_data, pt_data, ivfpq_cpu_data= map(get_data_list, file_names)

        
# Accuracy vs Time plot
plt.xscale("log")
plt.plot(ivfpq_reg_up_data[1], ivfpq_reg_up_data[2], label="IVFPQ GPU regular updates")
plt.plot(ivfpq_no_up_data[1], ivfpq_no_up_data[2], label="IVFPQ GPU dynamic update freq")
plt.plot(ivfpq_cpu_data[1], ivfpq_cpu_data[2], label="IVFPQ CPU dynamic update freq")
plt.plot(pt_data[1], pt_data[2], label="PyTorch GPU baseline")

plt.legend()
plt.xlabel("Time(s)")
plt.ylabel("Test accuracy")


plt.tight_layout()
plt.savefig("IVFPQ_AccVSt.png")

plt.clf()
plt.xscale("log")
plt.plot(ivfpq_reg_up_data[0], ivfpq_reg_up_data[2], label="IVFPQ GPU regular updates")
plt.plot(ivfpq_no_up_data[0], ivfpq_no_up_data[2], label="IVFPQ GPU dynamic update freq")
plt.plot(ivfpq_cpu_data[0], ivfpq_cpu_data[2], label="IVFPQ CPU dynamic update freq")
plt.plot(pt_data[0], pt_data[2], label="PyTorch GPU baseline")

plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Test accuracy")
plt.tight_layout()
plt.savefig("IVFPQ_AccVSIt.png")
