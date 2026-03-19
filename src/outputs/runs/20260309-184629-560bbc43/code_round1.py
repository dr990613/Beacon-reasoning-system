def compute_discounted_total(
    items: List[OrderItem],
    customer_tier: str,
    coupon_code: str | None = None,
) -> float:
    subtotal = compute_subtotal(items)
    discount_rate = compute_discount_rate(customer_tier, coupon_code)
    return round(subtotal * (1 - discount_rate), 2)