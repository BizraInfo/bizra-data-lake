#![warn(missing_docs)]
//! # BIZRA Sippar Protocol v1.0.0
//!
//! Regular number arithmetic for exact economic computation.
//!
//! ## Standing on Giants
//! - **Babylonian scribes (1900 BCE)**: Regular numbers (2,3,5-smooth) guarantee
//!   exact reciprocals in base-60. No floating-point drift. No rounding errors.
//! - **Dr. Daniel Mansfield (2021)**: Si.427 and Plimpton 322 reveal systematic
//!   application of exact arithmetic to surveying and administration.
//! - **BIZRA Constitution**: Adl (Justice) requires exact arithmetic. If 1/3
//!   cannot be represented exactly, then three-way splits are unjust by definition.
//!
//! ## Architectural Role
//! `RegularNumber` is the SAT-side (Rust) numeric type for all economic computation:
//! shard IDs, token amounts, tax rates, Gini coefficient inputs. The constraint
//! (only 2,3,5-smooth values allowed) is the capability (zero drift, exact division).
//!
//! ## The Compression Theorem
//! Constrain the operation space until the only reachable states are constitutional,
//! then every execution is a proof.

use std::fmt;

// ============================================================================
// CORE TYPE: RegularNumber
// ============================================================================

/// A 2,3,5-smooth integer: `value = 2^exp2 × 3^exp3 × 5^exp5`.
///
/// Guarantees exact reciprocals in base-60. Used for all economic
/// arithmetic in BIZRA where drift would violate constitutional Adl.
///
/// # Examples
/// ```
/// use bizra_sippar::RegularNumber;
///
/// // 60 = 2² × 3 × 5 (the Babylonian base)
/// let sixty = RegularNumber::from_u64(60).unwrap();
/// assert_eq!(sixty.exp2(), 2);
/// assert_eq!(sixty.exp3(), 1);
/// assert_eq!(sixty.exp5(), 1);
///
/// // 7 is irregular (prime, not in {2,3,5})
/// assert!(RegularNumber::from_u64(7).is_err());
///
/// // 1529 = 11 × 139 — the Si.427 mystery number (irregular witness)
/// assert!(!RegularNumber::is_regular(1529));
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegularNumber {
    exp2: u8,
    exp3: u8,
    exp5: u8,
    value: u64,
}

impl RegularNumber {
    /// Construct from prime factorization exponents.
    pub fn from_factors(exp2: u8, exp3: u8, exp5: u8) -> Result<Self, SipparError> {
        let value = checked_regular_product(exp2, exp3, exp5).ok_or(SipparError::Overflow)?;
        Ok(Self {
            exp2,
            exp3,
            exp5,
            value,
        })
    }

    /// Construct from a `u64`, validating that it is 2,3,5-smooth.
    ///
    /// Returns `Err(SipparError::IrregularFactor(p))` if `n` has a prime
    /// factor `p` outside {2, 3, 5}. This is the constitutional gate:
    /// irregular values are rejected at construction, not at use.
    pub fn from_u64(n: u64) -> Result<Self, SipparError> {
        if n == 0 {
            return Err(SipparError::Zero);
        }
        let mut remainder = n;
        let mut exp2: u8 = 0;
        let mut exp3: u8 = 0;
        let mut exp5: u8 = 0;

        while remainder.is_multiple_of(2) {
            remainder /= 2;
            exp2 += 1;
        }
        while remainder.is_multiple_of(3) {
            remainder /= 3;
            exp3 += 1;
        }
        while remainder.is_multiple_of(5) {
            remainder /= 5;
            exp5 += 1;
        }

        if remainder != 1 {
            return Err(SipparError::IrregularFactor(remainder));
        }
        Ok(Self {
            exp2,
            exp3,
            exp5,
            value: n,
        })
    }

    /// The underlying integer value.
    #[inline]
    pub fn value(&self) -> u64 {
        self.value
    }

    /// Exponent of 2 in the prime factorization.
    #[inline]
    pub fn exp2(&self) -> u8 {
        self.exp2
    }

    /// Exponent of 3 in the prime factorization.
    #[inline]
    pub fn exp3(&self) -> u8 {
        self.exp3
    }

    /// Exponent of 5 in the prime factorization.
    #[inline]
    pub fn exp5(&self) -> u8 {
        self.exp5
    }

    /// Check regularity without constructing. O(log n) trial division.
    pub fn is_regular(n: u64) -> bool {
        if n == 0 {
            return false;
        }
        let mut t = n;
        while t.is_multiple_of(2) {
            t /= 2;
        }
        while t.is_multiple_of(3) {
            t /= 3;
        }
        while t.is_multiple_of(5) {
            t /= 5;
        }
        t == 1
    }

    /// Exact reciprocal in base-60.
    ///
    /// Returns `(numerator, places)` where `1/self = numerator / 60^places`.
    /// Because `self` is regular and `60 = 2² × 3 × 5`, this division
    /// is exact — no remainder, no rounding, no drift.
    pub fn reciprocal(&self) -> (u64, u8) {
        if self.value == 1 {
            return (1, 0);
        }
        let k2 = self.exp2.div_ceil(2);
        let k = k2.max(self.exp3).max(self.exp5);
        let sixty_k = 60u64.pow(k as u32);
        (sixty_k / self.value, k)
    }

    /// Sexagesimal string representation of `1/self`.
    ///
    /// Uses standard Babylonian notation: `"0;20"` means 20/60 = 1/3.
    /// Multi-place fractions use comma separation: `"0;7,30"` means
    /// 7/60 + 30/3600 = 1/8.
    pub fn reciprocal_sexagesimal(&self) -> String {
        let (num, places) = self.reciprocal();
        if places == 0 {
            return "1".to_string();
        }
        let mut remaining = num;
        let mut digits = Vec::with_capacity(places as usize);
        for _ in 0..places {
            digits.push(remaining % 60);
            remaining /= 60;
        }
        digits.reverse();
        let parts: Vec<String> = digits.iter().map(|d| format!("{d}")).collect();
        format!("0;{}", parts.join(","))
    }

    /// Multiply two regular numbers. Result is guaranteed regular
    /// (closure under multiplication — the semigroup property).
    pub fn checked_mul(&self, other: &Self) -> Result<Self, SipparError> {
        let exp2 = self
            .exp2
            .checked_add(other.exp2)
            .ok_or(SipparError::Overflow)?;
        let exp3 = self
            .exp3
            .checked_add(other.exp3)
            .ok_or(SipparError::Overflow)?;
        let exp5 = self
            .exp5
            .checked_add(other.exp5)
            .ok_or(SipparError::Overflow)?;
        Self::from_factors(exp2, exp3, exp5)
    }

    /// Exact division of two regular numbers. Result is regular if and
    /// only if each exponent of the divisor is <= the dividend's.
    pub fn checked_div(&self, other: &Self) -> Result<Self, SipparError> {
        if other.value == 0 {
            return Err(SipparError::Zero);
        }
        let exp2 = self
            .exp2
            .checked_sub(other.exp2)
            .ok_or(SipparError::NotDivisible)?;
        let exp3 = self
            .exp3
            .checked_sub(other.exp3)
            .ok_or(SipparError::NotDivisible)?;
        let exp5 = self
            .exp5
            .checked_sub(other.exp5)
            .ok_or(SipparError::NotDivisible)?;
        Self::from_factors(exp2, exp3, exp5)
    }

    /// Generate the first `n` regular numbers in ascending order (Hamming sequence).
    pub fn first_n(n: usize) -> Vec<Self> {
        if n == 0 {
            return Vec::new();
        }
        let mut result: Vec<u64> = Vec::with_capacity(n);
        result.push(1);
        let (mut i2, mut i3, mut i5) = (0usize, 0usize, 0usize);
        while result.len() < n {
            let next2 = result[i2].saturating_mul(2);
            let next3 = result[i3].saturating_mul(3);
            let next5 = result[i5].saturating_mul(5);
            let next = next2.min(next3).min(next5);
            if next == u64::MAX {
                break;
            }
            result.push(next);
            if next == next2 {
                i2 += 1;
            }
            if next == next3 {
                i3 += 1;
            }
            if next == next5 {
                i5 += 1;
            }
        }
        result
            .into_iter()
            .filter_map(|v| Self::from_u64(v).ok())
            .collect()
    }
}

// ============================================================================
// DISPLAY & DEBUG
// ============================================================================

impl fmt::Debug for RegularNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Regular({}=2^{}*3^{}*5^{})",
            self.value, self.exp2, self.exp3, self.exp5
        )
    }
}

impl fmt::Display for RegularNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

// ============================================================================
// ERROR TYPE
// ============================================================================

/// Errors from the regular number constraint system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SipparError {
    /// Zero has no reciprocal in any base.
    Zero,
    /// Result exceeds `u64` representable range.
    Overflow,
    /// Input contains a prime factor outside {2, 3, 5}.
    IrregularFactor(u64),
    /// Division would produce a non-regular result.
    NotDivisible,
}

impl fmt::Display for SipparError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => write!(f, "zero has no reciprocal in Babylonian mathematics"),
            Self::Overflow => write!(f, "exceeds u64 regular number space"),
            Self::IrregularFactor(p) => write!(f, "irregular prime factor: {p}"),
            Self::NotDivisible => write!(f, "division produces non-regular result"),
        }
    }
}

impl std::error::Error for SipparError {}

/// Compute `2^a * 3^b * 5^c` with overflow checking.
fn checked_regular_product(a: u8, b: u8, c: u8) -> Option<u64> {
    2u64.checked_pow(a as u32)?
        .checked_mul(3u64.checked_pow(b as u32)?)?
        .checked_mul(5u64.checked_pow(c as u32)?)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_u64_regular_values() {
        for &n in &[
            1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50,
            54, 60,
        ] {
            assert!(RegularNumber::from_u64(n).is_ok(), "{n} should be regular");
        }
    }

    #[test]
    fn from_u64_irregular_values() {
        for &n in &[7, 11, 13, 14, 17, 19, 21, 22, 23, 26, 28, 29, 31, 1529] {
            assert!(
                RegularNumber::from_u64(n).is_err(),
                "{n} should be irregular"
            );
        }
    }

    #[test]
    fn from_u64_zero_rejected() {
        assert_eq!(RegularNumber::from_u64(0), Err(SipparError::Zero));
    }

    #[test]
    fn from_factors_sixty() {
        let r = RegularNumber::from_factors(2, 1, 1).unwrap();
        assert_eq!(r.value(), 60);
    }

    #[test]
    fn from_factors_one() {
        let r = RegularNumber::from_factors(0, 0, 0).unwrap();
        assert_eq!(r.value(), 1);
    }

    #[test]
    fn irregular_factor_diagnostic() {
        let err = RegularNumber::from_u64(1529).unwrap_err();
        assert_eq!(err, SipparError::IrregularFactor(1529));
    }

    #[test]
    fn si427_witness_must_be_irregular() {
        assert!(!RegularNumber::is_regular(1529));
    }

    #[test]
    fn reciprocal_of_one() {
        let one = RegularNumber::from_u64(1).unwrap();
        assert_eq!(one.reciprocal(), (1, 0));
        assert_eq!(one.reciprocal_sexagesimal(), "1");
    }

    #[test]
    fn reciprocal_of_two() {
        let two = RegularNumber::from_u64(2).unwrap();
        assert_eq!(two.reciprocal(), (30, 1));
        assert_eq!(two.reciprocal_sexagesimal(), "0;30");
    }

    #[test]
    fn reciprocal_of_three() {
        let three = RegularNumber::from_u64(3).unwrap();
        assert_eq!(three.reciprocal(), (20, 1));
        assert_eq!(three.reciprocal_sexagesimal(), "0;20");
    }

    #[test]
    fn reciprocal_of_four() {
        assert_eq!(
            RegularNumber::from_u64(4).unwrap().reciprocal_sexagesimal(),
            "0;15"
        );
    }

    #[test]
    fn reciprocal_of_five() {
        assert_eq!(
            RegularNumber::from_u64(5).unwrap().reciprocal_sexagesimal(),
            "0;12"
        );
    }

    #[test]
    fn reciprocal_of_six() {
        assert_eq!(
            RegularNumber::from_u64(6).unwrap().reciprocal_sexagesimal(),
            "0;10"
        );
    }

    #[test]
    fn reciprocal_of_eight() {
        let eight = RegularNumber::from_u64(8).unwrap();
        assert_eq!(eight.reciprocal(), (450, 2));
        assert_eq!(eight.reciprocal_sexagesimal(), "0;7,30");
    }

    #[test]
    fn reciprocal_of_nine() {
        assert_eq!(
            RegularNumber::from_u64(9).unwrap().reciprocal_sexagesimal(),
            "0;6,40"
        );
    }

    #[test]
    fn mul_closure() {
        let six = RegularNumber::from_u64(6).unwrap();
        let ten = RegularNumber::from_u64(10).unwrap();
        assert_eq!(six.checked_mul(&ten).unwrap().value(), 60);
    }

    #[test]
    fn div_exact() {
        let sixty = RegularNumber::from_u64(60).unwrap();
        let three = RegularNumber::from_u64(3).unwrap();
        assert_eq!(sixty.checked_div(&three).unwrap().value(), 20);
    }

    #[test]
    fn div_not_divisible() {
        let three = RegularNumber::from_u64(3).unwrap();
        let four = RegularNumber::from_u64(4).unwrap();
        assert_eq!(three.checked_div(&four), Err(SipparError::NotDivisible));
    }

    #[test]
    fn first_20_hamming_numbers() {
        let seq = RegularNumber::first_n(20);
        let values: Vec<u64> = seq.iter().map(|r| r.value()).collect();
        assert_eq!(
            values,
            vec![1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36]
        );
    }

    #[test]
    fn zakat_exact_fraction() {
        // Zakat = 2.5% = 1/40. 40 = 2^3 * 5 -> regular. 1/40 = 0;1,30
        let forty = RegularNumber::from_u64(40).unwrap();
        assert_eq!(forty.exp2(), 3);
        assert_eq!(forty.exp3(), 0);
        assert_eq!(forty.exp5(), 1);
        assert_eq!(forty.reciprocal_sexagesimal(), "0;1,30");
    }

    #[test]
    fn genesis_mystery_field() {
        // For the Genesis Receipt's 25:29 field:
        // The witness MUST be irregular (cannot be generated by regular arithmetic)
        let candidates = [1529u64, 7, 11, 13, 77, 91, 143, 1001];
        for &c in &candidates {
            assert!(
                !RegularNumber::is_regular(c),
                "witness {c} must be irregular"
            );
        }
    }
}
