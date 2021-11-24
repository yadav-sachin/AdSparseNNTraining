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
    "HNSW_regular_update_M32.txt",
    "HNSW_rebuild_no_update.txt",
    "PyTorchImplementation.txt",
    "SLIDE.txt"
    # "tf_gpu_256_Amazon670K.txt",
    # "tf_cpu_256_Amazon670K.txt",
    # "my.txt"
)

print("1")
hnsw_reg_up_data, hnsw_no_up_data, pt_data , slide_data = map(get_data_list, file_names)

        
print("2")
# Accuracy vs Time plot
plt.xscale("log")
plt.plot(hnsw_reg_up_data[1], hnsw_reg_up_data[2], label="HNSW CPU regular updates")
plt.plot(hnsw_no_up_data[1], hnsw_no_up_data[2], label="HNSW CPU dynamic update freq")
# plt.plot(tf_gpu_data[1], tf_gpu_data[2])
plt.plot(pt_data[1], pt_data[2], label="PyTorch GPU baseline")
# plt.plot(slide_data[1], slide_data[2], label="SLIDE")
# print(min(slide_data[1]))
print(min(pt_data[1]))
plt.legend()
plt.xlabel("Time(s)")
plt.ylabel("Test accuracy")
# plt.plot(tf_cpu_data[1], tf_cpu_data[2])
# plt.plot(my_data[1], my_data[2])
plt.tight_layout()
plt.savefig("h.png")

plt.clf()
plt.xscale("log")
plt.plot(hnsw_reg_up_data[0], hnsw_reg_up_data[2], label="HNSW CPU regular updates")
plt.plot(hnsw_no_up_data[0], hnsw_no_up_data[2], label="HNSW CPU dynamic update freq")
# plt.plot(tf_gpu_data[1], tf_gpu_data[2])
plt.plot(pt_data[0], pt_data[2], label="PyTorch GPU baseline")
# print(min(slide_data[1]))
print(min(pt_data[1]))
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Test accuracy")
# plt.plot(tf_cpu_data[1], tf_cpu_data[2])
# plt.plot(my_data[1], my_data[2])
plt.tight_layout()
plt.savefig("g.png")
