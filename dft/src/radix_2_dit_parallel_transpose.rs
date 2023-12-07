use p3_field::TwoAdicField;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::MatrixTranspose;
use p3_maybe_rayon::ParallelIterator;

use crate::{Radix2Dit, TwoAdicSubgroupDft};

#[derive(Default, Clone)]
pub struct Radix2DitParallelTranspose {}

/// A parallel FFT algorithm which uses a parallelizable cache-friendly matrix transposition implementation.
///
/// We compute a FFT algorithm over the columns of the original matrix.
impl<F: TwoAdicField> TwoAdicSubgroupDft<F> for Radix2DitParallelTranspose {
    fn dft_batch(&self, mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let radix2_dit = Radix2Dit {};

        // Transpose the matrix
        let mut mat = mat.transpose();

        // Compute FFT for each row (original columns) in parallel
        mat.par_row_chunks_mut(1).for_each(|row| {
            let fft_result = radix2_dit.dft(row.values.to_vec());
            row.values.copy_from_slice(&fft_result);
        });

        mat.transpose()
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_goldilocks::Goldilocks;

    use super::*;
    use crate::testing::*;

    #[test]
    fn dft_matches_naive_one_dimensional() {
        test_one_dim_dft_matches_naive::<BabyBear, Radix2DitParallelTranspose>();
    }

    #[test]
    fn dft_matches_naive() {
        test_dft_matches_naive::<BabyBear, Radix2DitParallelTranspose>();
    }

    #[test]
    fn coset_dft_matches_naive() {
        test_coset_dft_matches_naive::<BabyBear, Radix2DitParallelTranspose>();
    }

    #[test]
    fn idft_matches_naive() {
        test_idft_matches_naive::<Goldilocks, Radix2DitParallelTranspose>();
    }

    #[test]
    fn lde_matches_naive() {
        test_lde_matches_naive::<BabyBear, Radix2DitParallelTranspose>();
    }

    #[test]
    fn coset_lde_matches_naive() {
        test_coset_lde_matches_naive::<BabyBear, Radix2DitParallelTranspose>();
    }

    #[test]
    fn dft_idft_consistency() {
        test_dft_idft_consistency::<BabyBear, Radix2DitParallelTranspose>();
    }
}
