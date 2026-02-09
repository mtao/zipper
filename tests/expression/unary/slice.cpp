

#include <iostream>
#include <zipper/Matrix.hpp>
#include <zipper/Vector.hpp>
#include <zipper/VectorBase.hxx>
#include <zipper/detail/extents/dynamic_extents_indices.hpp>
#include <zipper/detail/extents/swizzle_extents.hpp>
#include <zipper/types.hpp>
#include <zipper/expression/nullary/Constant.hpp>
#include <zipper/expression/nullary/Random.hpp>

#include "../../catch_include.hpp"
#include "../../fmt_include.hpp"

namespace {
void print(zipper::concepts::Matrix auto const& M) {
    for (zipper::index_type j = 0; j < M.extent(0); ++j) {
        for (zipper::index_type k = 0; k < M.extent(1); ++k) {
            std::cout << M(j, k) << " ";
        }
        std::cout << std::endl;
    }
}
void print(zipper::concepts::Array auto const& M) {
    for (zipper::index_type j = 0; j < M.extent(0); ++j) {
        for (zipper::index_type k = 0; k < M.extent(1); ++k) {
            std::cout << M(j, k) << " ";
        }
        std::cout << std::endl;
    }
}
void print(zipper::concepts::Vector auto const& M) {
    for (zipper::index_type j = 0; j < M.extent(0); ++j) {
        std::cout << M(j) << " ";
    }
    std::cout << std::endl;
}
}  // namespace

// === From expression/test_slicing.cpp ===

TEST_CASE("test_matrix_slice_shapes", "[extents][matrix][slice]") {
    //fmt::print("Manipulating MN: \n");
    zipper::MatrixBase RN(zipper::expression::nullary::normal_random<double>(
        zipper::extents<4, 5>{}, 0, 20));
    zipper::Matrix MN = RN;
    {
        auto full_slice =
            MN.slice<zipper::full_extent_t, zipper::full_extent_t>();

        using ST = std::decay_t<decltype(full_slice.expression())>;
        static_assert(ST::actionable_indices.size() == 2);
        static_assert(ST::actionable_indices[0] == 0);
        static_assert(ST::actionable_indices[1] == 1);

        static_assert(std::is_same_v<ST::extents_type, zipper::extents<4, 5>>);

        REQUIRE(3 == full_slice.expression().get_index<0>(3, 4));
        REQUIRE(4 == full_slice.expression().get_index<1>(3, 4));
    }
    {
        auto slice = MN.slice<std::integral_constant<zipper::index_type, 1>,
                              zipper::full_extent_t>();
        using ST = std::decay_t<decltype(slice.expression())>;
        using T = ST::slice_storage_type;

        static_assert(std::is_same_v<zipper::full_extent_t,
                                     std::tuple_element_t<1, T>::type>);

        static_assert(
            std::is_same_v<std::integral_constant<zipper::index_type, 1>,
                           std::tuple_element_t<0, T>::type>);

        static_assert(zipper::concepts::IndexSlice<std::tuple_element_t<0, T>::type>);
        static_assert(zipper::concepts::IndexSlice<std::tuple_element_t<1, T>::type>);

        static_assert(zipper::concepts::Index<std::tuple_element_t<0, T>::type>);
        static_assert(!zipper::concepts::Index<std::tuple_element_t<1, T>::type>);

        static_assert(std::is_same_v<zipper::full_extent_t,
                                     std::tuple_element_t<1, T>::type>);
        static_assert(ST::actionable_indices.size() == 2);
        static_assert(ST::actionable_indices[0] == std::dynamic_extent);
        static_assert(ST::actionable_indices[1] == 0);
        static_assert(ST::extents_type::rank() == 1);
        static_assert(ST::extents_type::static_extent(0) == 5);
        static_assert(std::is_same_v<ST::extents_type, zipper::extents<5>>);

        REQUIRE(1 == slice.expression().get_index<0>(4));
        REQUIRE(4 == slice.expression().get_index<1>(4));
    }
    {
        auto slice = MN.slice<zipper::full_extent_t,
                              std::integral_constant<zipper::index_type, 1>>();
        using ST = std::decay_t<decltype(slice.expression())>;
        using T = ST::slice_storage_type;

        static_assert(std::is_same_v<zipper::full_extent_t,
                                     std::tuple_element_t<0, T>::type>);

        static_assert(
            std::is_same_v<std::integral_constant<zipper::index_type, 1>,
                           std::tuple_element_t<1, T>::type>);

        static_assert(zipper::concepts::IndexSlice<std::tuple_element_t<0, T>::type>);
        static_assert(zipper::concepts::IndexSlice<std::tuple_element_t<1, T>::type>);

        static_assert(zipper::concepts::Index<std::tuple_element_t<1, T>::type>);
        static_assert(!zipper::concepts::Index<std::tuple_element_t<0, T>::type>);

        static_assert(std::is_same_v<zipper::full_extent_t,
                                     std::tuple_element_t<0, T>::type>);
        static_assert(ST::actionable_indices.size() == 2);
        static_assert(ST::actionable_indices[1] == std::dynamic_extent);
        static_assert(ST::actionable_indices[0] == 0);
        static_assert(std::is_same_v<ST::extents_type, zipper::extents<4>>);

        REQUIRE(1 == slice.expression().get_index<1>(4));
        REQUIRE(4 == slice.expression().get_index<0>(4));
    }
}

TEST_CASE("test_matrix_slicing", "[extents][matrix][slice]") {
    //fmt::print("Manipulating MN: \n");
    zipper::MatrixBase RN(zipper::expression::nullary::normal_random<double>(
        zipper::extents<4, 4>{}, 0, 20));

    zipper::Matrix MN = RN;
    zipper::Matrix<double, std::dynamic_extent, std::dynamic_extent> MND = MN;
    auto full_slice = MN.slice<zipper::full_extent_t, zipper::full_extent_t>();
    auto full_sliceD =
        MND.slice<zipper::full_extent_t, zipper::full_extent_t>();

    REQUIRE(MN.extents() == full_slice.extents());
    REQUIRE(full_sliceD.extents() == full_slice.extents());
    //fmt::print("MN\n");
    print(MN);
    //fmt::print("full slice\n");
    print(MN.slice<zipper::full_extent_t, zipper::full_extent_t>());
    //fmt::print("full slice\n");
    print(MN.slice<zipper::full_extent_t, zipper::full_extent_t>());
    //fmt::print("are same\n");
    print(
        (MN.slice<zipper::full_extent_t, zipper::full_extent_t>().as_array() ==
         MN.as_array()));

    CHECK((MN.slice<zipper::full_extent_t, zipper::full_extent_t>() == MN));
    CHECK((MN.slice(zipper::full_extent_t{}, zipper::full_extent_t{}) == MN));
    CHECK((MND.slice<zipper::full_extent_t, zipper::full_extent_t>() == MN));
    CHECK((MND.slice(zipper::full_extent_t{}, zipper::full_extent_t{}) == MN));

    // zipper::slice(zipper::index_type(1));
    {
        auto S = MN.slice(zipper::slice(1, MN.extent(0) - 1),
                          zipper::full_extent_t{});

        auto SD = MND.slice(zipper::slice(1, MND.extent(0) - 1),
                            zipper::full_extent_t{});

        REQUIRE(S.extent(0) == MN.extent(0) - 1);
        REQUIRE(S.extent(1) == MN.extent(1));

        REQUIRE(SD.extents() == S.extents());
        for (const auto& [a, b] :
             zipper::utils::extents::all_extents_indices(S.extents())) {
            static_assert(std::is_integral_v<std::decay_t<decltype(a)>>);
            static_assert(std::is_integral_v<std::decay_t<decltype(b)>>);
            CHECK(S(a, b) == MN(a + 1, b));
            CHECK(SD(a, b) == S(a, b));
    }
    }



    auto slice = MN.slice<std::integral_constant<zipper::index_type, 1>,
                          zipper::full_extent_t>();
    auto sliceD = MND.slice<std::integral_constant<zipper::index_type, 1>,
                            zipper::full_extent_t>();
    REQUIRE(slice.extent(0) == 4);
    REQUIRE(slice.extents().rank() == 1);
    CHECK(slice(0) == MN(1, 0));
    CHECK(slice(1) == MN(1, 1));
    CHECK(slice(2) == MN(1, 2));
    CHECK(slice(3) == MN(1, 3));
    slice(0) = 2.0;
    slice(1) = 100000;
    slice(2) = 104000;
    slice(3) = 100003;

    REQUIRE(sliceD.extent(0) == 4);
    REQUIRE(sliceD.extents().rank() == 1);
    CHECK(sliceD(0) == MND(1, 0));
    CHECK(sliceD(1) == MND(1, 1));
    CHECK(sliceD(2) == MND(1, 2));
    CHECK(sliceD(3) == MND(1, 3));
    sliceD(0) = 2.0;
    sliceD(1) = 100000;
    sliceD(2) = 104000;
    sliceD(3) = 100003;

    CHECK(2.0 == MN(1, 0));
    CHECK(100000 == MN(1, 1));
    CHECK(104000 == MN(1, 2));
    CHECK(100003 == MN(1, 3));
    CHECK(2.0 == MND(1, 0));
    CHECK(100000 == MND(1, 1));
    CHECK(104000 == MND(1, 2));
    CHECK(100003 == MND(1, 3));

    slice = zipper::expression::nullary::Constant<double>(3.4);
    sliceD = zipper::expression::nullary::Constant<double>(3.4);
    CHECK(3.4 == MN(1, 0));
    CHECK(3.4 == MN(1, 1));
    CHECK(3.4 == MN(1, 2));
    CHECK(3.4 == MN(1, 3));
    CHECK(3.4 == MND(1, 0));
    CHECK(3.4 == MND(1, 1));
    CHECK(3.4 == MND(1, 2));
    CHECK(3.4 == MND(1, 3));

    MN.row<std::integral_constant<zipper::index_type, 2>>() = slice;
    MND.row<std::integral_constant<zipper::index_type, 2>>() = slice;

    CHECK(3.4 == MN(2, 0));
    CHECK(3.4 == MN(2, 1));
    CHECK(3.4 == MN(2, 2));
    CHECK(3.4 == MN(2, 3));

    CHECK(3.4 == MND(2, 0));
    CHECK(3.4 == MND(2, 1));
    CHECK(3.4 == MND(2, 2));
    CHECK(3.4 == MND(2, 3));
    MN.col<std::integral_constant<zipper::index_type, 2>>() = slice;

    CHECK(3.4 == MN(0, 2));
    CHECK(3.4 == MN(1, 2));
    CHECK(3.4 == MN(2, 2));
    CHECK(3.4 == MN(3, 2));
    MN.col(2) = slice * 2;

    CHECK(2 * 3.4 == MN(0, 2));
    CHECK(2 * 3.4 == MN(1, 2));
    CHECK(2 * 3.4 == MN(2, 2));
    CHECK(2 * 3.4 == MN(3, 2));

    MN = RN;

    CHECK(slice(0) == MN(1, 0));
    CHECK(slice(1) == MN(1, 1));
    CHECK(slice(2) == MN(1, 2));
    CHECK(slice(3) == MN(1, 3));
    return;
    auto diag = MN.diagonal();
    REQUIRE(diag.extents() == zipper::create_dextents(4));
    CHECK(diag(0) == MN(0, 0));
    CHECK(diag(1) == MN(1, 1));
    CHECK(diag(2) == MN(2, 2));
    CHECK(diag(3) == MN(3, 3));

    auto slice2 = MN(std::integral_constant<zipper::index_type, 3>{},
                     zipper::full_extent_t{});
    CHECK(slice2(0) == MN(3, 0));
    CHECK(slice2(1) == MN(3, 1));
    CHECK(slice2(2) == MN(3, 2));
    CHECK(slice2(3) == MN(3, 3));

    zipper::Matrix NN = MN;
    slice2 = 100 * slice;

    CHECK(slice2(0) == 100 * NN(1, 0));
    CHECK(slice2(1) == 100 * NN(1, 1));
    CHECK(slice2(2) == 100 * NN(1, 2));
    CHECK(slice2(3) == 100 * NN(1, 3));
    NN = MN;
    auto slice3 = MN(2, zipper::full_extent_t{});
    slice3 = 50 * slice2;
    CHECK(slice3(0) == 50 * NN(1, 0));
    CHECK(slice3(1) == 50 * NN(1, 1));
    CHECK(slice3(2) == 50 * NN(1, 2));
    CHECK(slice3(3) == 50 * 100 * NN(1, 3));

    auto slice4 = MN(3, zipper::full_extent_t{});
    CHECK(slice4(0) == MN(3, 0));
    CHECK(slice4(1) == MN(3, 1));
    CHECK(slice4(2) == MN(3, 2));
    CHECK(slice4(3) == MN(3, 3));
}

TEST_CASE("test_vector_slice_assignment", "[extents][vector][slice]") {
    zipper::Vector<double, 4> x;

    x(3) = 2;

    x.head<3>() = zipper::expression::nullary::Constant<double>(1);
    CHECK(x(0) == 1);
    CHECK(x(1) == 1);
    CHECK(x(2) == 1);
    CHECK(x(3) == 2);
}

TEST_CASE("test_partial_slice", "[extents][tensor][slice]") {
    //fmt::print("Manipulating MN: \n");
}

// test_span_array_access uses std::array/std::vector which already satisfy IndexSlice.
// test_span_view_access uses Vector<index_type> which satisfies IndexSlice
// via the IsZipperBase specialization in IndexSlice.hpp.
TEST_CASE("test_span_array_access", "[vector][storage][dense][span]") {
    {
        zipper::Vector<double, 3> x = {1, 2, 3};

        std::array<zipper::index_type, 2> a{{0, 2}};
        auto s = x(a);
        REQUIRE(s.extents().rank() == 1);
        CHECK(s(0) == 1);
        CHECK(s(1) == 3);
    }
    {
        zipper::Vector<double, std::dynamic_extent> x = {1, 2, 3};

        std::vector<zipper::index_type> a{{0,2}};
        auto s = x(a);
        REQUIRE(s.extents().rank() == 1);
        CHECK(s(0) == 1);
        CHECK(s(1) == 3);



    }
}
TEST_CASE("test_span_view_access", "[vector][storage][dense][span]") {
    {
        zipper::Vector<double, 3> x = {1, 2, 3};

        zipper::Vector<zipper::index_type,2> a = {{0,2}};
        auto s = x(a);
        REQUIRE(s.extents().rank() == 1);
        CHECK(s(0) == 1);
        CHECK(s(1) == 3);
    }
    {
        zipper::Vector<double, std::dynamic_extent> x = {1, 2, 3};

        zipper::Vector<zipper::index_type,std::dynamic_extent> a = {{0,2}};
        auto s = x(a);
        REQUIRE(s.extents().rank() == 1);
        CHECK(s(0) == 1);
        CHECK(s(1) == 3);



    }
    {
        zipper::Matrix<double, 2,std::dynamic_extent> V = {{1, 2, 3},{0,2,1}};

        zipper::Vector<zipper::index_type,2> a = {{0,2}};
        auto s = V(0,a);
        REQUIRE(s.extents().rank() == 1);
        CHECK(s(0) == 1);
        CHECK(s(1) == 3);
        auto r = V(1,a);
        REQUIRE(r.extents().rank() == 1);
        CHECK(r(0) == 0);
        CHECK(r(1) == 1);
    }
}

// === From expression/test_vector.cpp (head/tail tests) ===

TEST_CASE("test_head_tail", "[vector][homogeneous]") {
    zipper::Vector a = zipper::expression::nullary::uniform_random<double>(
        zipper::extents<3>{});
    zipper::Vector b = zipper::expression::nullary::uniform_random<double>(
        zipper::create_dextents(3));

    auto c = a.head<2>();
    REQUIRE(c.extent(0) == 2);
    CHECK(c(0) == a(0));
    CHECK(c(1) == a(1));
    auto d = a.tail<2>();
    REQUIRE(d.extent(0) == 2);
    CHECK(d(0) == a(1));
    CHECK(d(1) == a(2));

    auto e = b.head<2>();
    REQUIRE(c.extent(0) == 2);
    CHECK(e(0) == b(0));
    CHECK(e(1) == b(1));
    auto f = b.tail<2>();
    REQUIRE(f.extent(0) == 2);
    CHECK(f(0) == b(1));
    CHECK(f(1) == b(2));
}

// === From test_vector.cpp (vector slicing tests) ===

using namespace zipper;
TEST_CASE("test_vector_slicing", "[vector][storage][dense]") {
  Vector<double, 5> a{{0, 2, 4, 6, 8}};
  Vector<double, std::dynamic_extent> b{{1, 3, 5, 7, 9}};

  {
    auto av = a.head<2>();
    auto bv = b.head<2>();
    REQUIRE(av.extents() == create_dextents(2));
    REQUIRE(bv.extents() == create_dextents(2));
    CHECK(av(0) == 0);
    CHECK(av(1) == 2);
    CHECK(bv(0) == 1);
    CHECK(bv(1) == 3);
  }

  {
    auto av = a.head(2);
    auto bv = b.head(2);
    REQUIRE(av.extents() == create_dextents(2));
    REQUIRE(bv.extents() == create_dextents(2));
    CHECK(av(0) == 0);
    CHECK(av(1) == 2);
    CHECK(bv(0) == 1);
    CHECK(bv(1) == 3);
  }
  {
    auto av = a.tail<2>();
    auto bv = b.tail<2>();
    REQUIRE(av.extents() == create_dextents(2));
    REQUIRE(bv.extents() == create_dextents(2));
    CHECK(av(0) == 6);
    CHECK(av(1) == 8);
    CHECK(bv(0) == 7);
    CHECK(bv(1) == 9);
  }

  {
    auto av = a.tail(2);
    auto bv = b.tail(2);
    REQUIRE(av.extents() == create_dextents(2));
    REQUIRE(bv.extents() == create_dextents(2));
    CHECK(av(0) == 6);
    CHECK(av(1) == 8);
    CHECK(bv(0) == 7);
    CHECK(bv(1) == 9);
  }
  {
    auto av = a.segment<1, 2>();
    auto bv = b.segment<1, 2>();
    REQUIRE(av.extents() == create_dextents(2));
    REQUIRE(bv.extents() == create_dextents(2));
    CHECK(av(0) == 2);
    CHECK(av(1) == 4);
    CHECK(bv(0) == 3);
    CHECK(bv(1) == 5);
  }

  {
    auto av = a.segment<2>(1);
    auto bv = b.segment<2>(1);
    REQUIRE(av.extents() == create_dextents(2));
    REQUIRE(bv.extents() == create_dextents(2));
    CHECK(av(0) == 2);
    CHECK(av(1) == 4);
    CHECK(bv(0) == 3);
    CHECK(bv(1) == 5);
  }
  {
    auto av = a.segment(1, 2);
    auto bv = b.segment(1, 2);
    REQUIRE(av.extents() == create_dextents(2));
    REQUIRE(bv.extents() == create_dextents(2));
    CHECK(av(0) == 2);
    CHECK(av(1) == 4);
    CHECK(bv(0) == 3);
    CHECK(bv(1) == 5);
  }
}

// === From test_matrix.cpp (block tests: topRows, bottomRows, leftCols, rightCols) ===

TEST_CASE("test_blocks", "[matrix][storage][dense]") {
  zipper::Matrix<zipper::index_type, 3, 6> C;
  zipper::Matrix<zipper::index_type, std::dynamic_extent, std::dynamic_extent>
      R(3, 6);

  {
    for (zipper::index_type j = 0; j < 3; ++j) {
      for (zipper::index_type k = 0; k < 6; ++k) {
        C(j, k) = j;
        R(j, k) = j;
      }
    }
    auto CR = C.topRows(2);
    auto CC = C.topRows(zipper::static_index_t<2>{});
    auto RR = R.topRows(2);
    auto RC = R.topRows(zipper::static_index_t<2>{});
    static_assert(CC.static_extent(0) == 2);
    static_assert(CR.static_extent(0) == std::dynamic_extent);
    static_assert(RC.static_extent(0) == 2);
    static_assert(RR.static_extent(0) == std::dynamic_extent);
    REQUIRE(CR.extents() == zipper::create_dextents(2, 6));
    REQUIRE(CR.extents() == CC.extents());
    REQUIRE(CR.extents() == CC.extents());
    REQUIRE(RR.extents() == CC.extents());
    REQUIRE(RR.extents() == CC.extents());
    for (zipper::index_type j = 0; j < 2; ++j) {
      for (zipper::index_type k = 0; k < 6; ++k) {
        CHECK(CC(j, k) == j);
        CHECK(CR(j, k) == j);
        CHECK(RC(j, k) == j);
        CHECK(RR(j, k) == j);
      }
    }
  }

  {
    for (zipper::index_type j = 0; j < 3; ++j) {
      for (zipper::index_type k = 0; k < 6; ++k) {
        C(j, k) = k;
        R(j, k) = k;
      }
    }
    auto CR = C.leftCols(5);
    auto CC = C.leftCols(zipper::static_index_t<5>{});
    auto RR = R.leftCols(5);
    auto RC = R.leftCols(zipper::static_index_t<5>{});
    static_assert(CC.static_extent(1) == 5);
    static_assert(CR.static_extent(1) == std::dynamic_extent);
    static_assert(RC.static_extent(1) == 5);
    static_assert(RR.static_extent(1) == std::dynamic_extent);
    REQUIRE(CR.extents() == zipper::create_dextents(3, 5));
    REQUIRE(CR.extents() == CC.extents());
    REQUIRE(CR.extents() == CC.extents());
    REQUIRE(RR.extents() == CC.extents());
    REQUIRE(RR.extents() == CC.extents());
    for (zipper::index_type j = 0; j < 3; ++j) {
      for (zipper::index_type k = 0; k < 5; ++k) {
        CHECK(CC(j, k) == k);
        CHECK(CR(j, k) == k);
        CHECK(RC(j, k) == k);
        CHECK(RR(j, k) == k);
      }
    }
  }
  {
    for (zipper::index_type j = 0; j < 3; ++j) {
      for (zipper::index_type k = 0; k < 6; ++k) {
        C(j, k) = j;
        R(j, k) = j;
      }
    }
    auto CR = C.bottomRows(2);
    auto CC = C.bottomRows(zipper::static_index_t<2>{});
    auto RR = R.bottomRows(2);
    auto RC = R.bottomRows(zipper::static_index_t<2>{});

    using CRT = std::decay_t<decltype(CR)>;
    using CRT_R =
        CRT::expression_type::traits::extents_helper::single_slice_helper<0>;
    using CRT_C =
        CRT::expression_type::traits::extents_helper::single_slice_helper<1>;

    static_assert(
        std::is_same_v<CRT_R::type,
                       zipper::slice_t<zipper::index_type, zipper::index_type,
                                       zipper::index_type>>);

    static_assert(std::is_same_v<CRT_C::type, zipper::full_extent_t>);
    // static_assert(
    static_assert(CC.static_extent(0) == 2);
    static_assert(CR.static_extent(0) == std::dynamic_extent);
    static_assert(RC.static_extent(0) == std::dynamic_extent);
    static_assert(RR.static_extent(0) == std::dynamic_extent);
    REQUIRE(CR.extents() == zipper::create_dextents(2, 6));
    REQUIRE(CR.extents() == CC.extents());
    REQUIRE(CR.extents() == CC.extents());
    REQUIRE(RR.extents() == CC.extents());
    REQUIRE(RR.extents() == CC.extents());
    for (zipper::index_type j = 0; j < 2; ++j) {
      for (zipper::index_type k = 0; k < 6; ++k) {
        CHECK(CC(j, k) == j + 1);
        CHECK(CR(j, k) == j + 1);
        CHECK(RC(j, k) == j + 1);
        CHECK(RR(j, k) == j + 1);
      }
    }
  }

  {
    for (zipper::index_type j = 0; j < 3; ++j) {
      for (zipper::index_type k = 0; k < 6; ++k) {
        C(j, k) = k;
        R(j, k) = k;
      }
    }
    auto CR = C.rightCols(5);
    auto CC = C.rightCols(zipper::static_index_t<5>{});
    auto RR = R.rightCols(5);
    auto RC = R.rightCols(zipper::static_index_t<5>{});
    static_assert(CC.static_extent(1) == 5);
    static_assert(CR.static_extent(1) == std::dynamic_extent);
    static_assert(RC.static_extent(1) == std::dynamic_extent);
    static_assert(RR.static_extent(1) == std::dynamic_extent);
    REQUIRE(CR.extents() == zipper::create_dextents(3, 5));
    REQUIRE(CR.extents() == CC.extents());
    REQUIRE(CR.extents() == CC.extents());
    REQUIRE(RR.extents() == CC.extents());
    REQUIRE(RR.extents() == CC.extents());
    for (zipper::index_type j = 0; j < 3; ++j) {
      for (zipper::index_type k = 0; k < 5; ++k) {
        CHECK(CC(j, k) == k + 1);
        CHECK(CR(j, k) == k + 1);
        CHECK(RC(j, k) == k + 1);
        CHECK(RR(j, k) == k + 1);
      }
    }
  }
}
