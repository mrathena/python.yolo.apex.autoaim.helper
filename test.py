


def greatestCommonDivisor(x: int, y: int):
    a, b = x, y
    divisor = 1  # 创建t作为最大公约数的载体
    for i in range(2, min(a, b)):
        while (a % i == 0 and b % i == 0):
            divisor *= i  # 所有公约数累乘起来
            a /= i
            b /= i
    return divisor


def lowestCommonMultiple(x: int, y: int):
    m, n = x, y
    k = m * n  # k存储两数的乘积
    if m < n:  # 比较两个数的大小，使得m中存储大数，n中存储小数
        temp = m
        m = n
        n = temp
    b = m % n  # b存储m除以n的余数
    while b != 0:
        m = n  # 原来的小数作为下次运算时的大数
        n = b  # 将上一次的余数作为下次相除时的小数
        b = m % n
    result = k // n  # 两数乘积除以最大公约数即为它们的最小公倍数
    return result


if __name__ == '__main__':

    print(greatestCommonDivisor(0, 3))
    print(lowestCommonMultiple(2, 0))


