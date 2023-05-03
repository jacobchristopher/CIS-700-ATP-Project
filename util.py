
def list_average(lists):
    
    averages1 = []
    for values in zip(*lists[0]):
        average = sum(values) / len(values)
        averages1.append(average)

    averages2 = []
    for values in zip(*lists[1]):
        average = sum(values) / len(values)
        averages2.append(average)

    return averages1, averages2
