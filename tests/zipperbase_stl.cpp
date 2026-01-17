
#include <zipper/Matrix.hpp>
#include <zipper/VectorBase.hpp>

TEST_CASE("stl_storage_zipper_bases", "[data]") {
  {
    std::vector<double> x = {0, 1, 2};
    auto storage = zipper::storage::get_non_owning_stl_storage(x);
    zipper::VectorBase X = storage;
    CHECK(X(0) == 0);
    CHECK(X(1) == 1);
    CHECK(X(2) == 2);

    CHECK(X.view().coeff(0) == 0);
    CHECK(X.view().coeff(1) == 1);
    CHECK(X.view().coeff(2) == 2);
    CHECK(X.view().coeff_ref(0) == 0);
    CHECK(X.view().coeff_ref(1) == 1);
    CHECK(X.view().coeff_ref(2) == 2);
    CHECK(X.view().const_coeff_ref(0) == 0);
    CHECK(X.view().const_coeff_ref(1) == 1);
    CHECK(X.view().const_coeff_ref(2) == 2);

    X(0) = 4;
    X(1) = 5;
    X(2) = 6;
    CHECK(X(0) == 4);
    CHECK(X(1) == 5);
    CHECK(X(2) == 6);

    using T = std::decay_t<decltype(X)>::view_type;
    using traits = zipper::views::detail::ViewTraits<T>;

    static_assert(!traits::is_const);
    static_assert(traits::is_writable);
    X = {0, 4, 5, 6};
    CHECK(x.size() == 4);
    CHECK(X.size() == 4);
    CHECK(X.rows() == 4);

    CHECK(X(0) == 0);
    CHECK(X(1) == 4);
    CHECK(X(2) == 5);
    CHECK(X(3) == 6);
  }

  {
    std::array<double, 3> x{{0, 1, 2}};
    auto storage = zipper::storage::get_non_owning_stl_storage(x);
    zipper::VectorBase X = storage;
    CHECK(X(0) == 0);
    CHECK(X(1) == 1);
    CHECK(X(2) == 2);

    CHECK(X.view().coeff(0) == 0);
    CHECK(X.view().coeff(1) == 1);
    CHECK(X.view().coeff(2) == 2);
    CHECK(X.view().coeff_ref(0) == 0);
    CHECK(X.view().coeff_ref(1) == 1);
    CHECK(X.view().coeff_ref(2) == 2);
    CHECK(X.view().const_coeff_ref(0) == 0);
    CHECK(X.view().const_coeff_ref(1) == 1);
    CHECK(X.view().const_coeff_ref(2) == 2);

    X(0) = 4;
    X(1) = 5;
    X(2) = 6;

    CHECK(X(0) == 4);
    CHECK(X(1) == 5);
    CHECK(X(2) == 6);

    X = {0, 4, 5};
    CHECK(x.size() == 3);
    CHECK(X.size() == 3);
    CHECK(X.rows() == 3);

    CHECK(X(0) == 0);
    CHECK(X(1) == 4);
    CHECK(X(2) == 5);
  }
  {
    std::vector<std::array<double, 3>> x{{0, 1, 2}, {3, 4, 5}};

    auto storage = zipper::storage::get_non_owning_stl_storage(x);
    zipper::MatrixBase X = storage;
    CHECK(X(0, 0) == 0);
    CHECK(X(0, 1) == 1);
    CHECK(X(0, 2) == 2);
    CHECK(X(1, 0) == 3);
    CHECK(X(1, 1) == 4);
    CHECK(X(1, 2) == 5);

    CHECK(X.view().coeff(0, 0) == 0);
    CHECK(X.view().coeff(0, 1) == 1);
    CHECK(X.view().coeff(0, 2) == 2);
    CHECK(X.view().coeff(1, 0) == 3);
    CHECK(X.view().coeff(1, 1) == 4);
    CHECK(X.view().coeff(1, 2) == 5);
    CHECK(X.view().coeff_ref(0, 0) == 0);
    CHECK(X.view().coeff_ref(0, 1) == 1);
    CHECK(X.view().coeff_ref(0, 2) == 2);
    CHECK(X.view().coeff_ref(1, 0) == 3);
    CHECK(X.view().coeff_ref(1, 1) == 4);
    CHECK(X.view().coeff_ref(1, 2) == 5);
    CHECK(X.view().const_coeff_ref(0, 0) == 0);
    CHECK(X.view().const_coeff_ref(0, 1) == 1);
    CHECK(X.view().const_coeff_ref(0, 2) == 2);
    CHECK(X.view().const_coeff_ref(1, 0) == 3);
    CHECK(X.view().const_coeff_ref(1, 1) == 4);
    CHECK(X.view().const_coeff_ref(1, 2) == 5);

    X(0, 0) = 14;
    X(0, 1) = 15;
    X(0, 2) = 16;
    X(1, 0) = 24;
    X(1, 1) = 25;
    X(1, 2) = 26;

    CHECK(x[0][0] == 14);
    CHECK(x[0][1] == 15);
    CHECK(x[0][2] == 16);
    CHECK(x[1][0] == 24);
    CHECK(x[1][1] == 25);
    CHECK(x[1][2] == 26);

    CHECK(X(0, 0) == 14);
    CHECK(X(0, 1) == 15);
    CHECK(X(0, 2) == 16);
    CHECK(X(1, 0) == 24);
    CHECK(X(1, 1) == 25);
    CHECK(X(1, 2) == 26);

    zipper::Matrix<double, 2, 3> B;
    CHECK(B.extents() == zipper::create_dextents(2, 3));
    CHECK(X.extents() == zipper::create_dextents(2, 3));
    B.row(0) = {0, 1, 2};
    B.row(1) = {3, 4, 5};
    X = B;
    CHECK(X.extents() == zipper::create_dextents(2, 3));
    CHECK((X == B));
    zipper::Matrix<double, 3, 3> C;
    CHECK(C.extents() == zipper::create_dextents(3, 3));
    C.row(0) = {0, 1, 2};
    C.row(1) = {3, 4, 5};
    C.row(2) = {6, 7, 8};
    X = C;
    CHECK(X.extents() == zipper::create_dextents(3, 3));
    CHECK((X == C));
  }

  {
    std::array<std::vector<double>, 2> x{{{0, 1, 2}, {3, 4, 5}}};

    auto storage = zipper::storage::get_non_owning_stl_storage(x);
    zipper::MatrixBase X = storage;
    CHECK(X(0, 0) == 0);
    CHECK(X(0, 1) == 1);
    CHECK(X(0, 2) == 2);
    CHECK(X(1, 0) == 3);
    CHECK(X(1, 1) == 4);
    CHECK(X(1, 2) == 5);

    X(0, 0) = 14;
    X(0, 1) = 15;
    X(0, 2) = 16;
    X(1, 0) = 24;
    X(1, 1) = 25;
    X(1, 2) = 26;

    CHECK(x[0][0] == 14);
    CHECK(x[0][1] == 15);
    CHECK(x[0][2] == 16);
    CHECK(x[1][0] == 24);
    CHECK(x[1][1] == 25);
    CHECK(x[1][2] == 26);

    CHECK(X(0, 0) == 14);
    CHECK(X(0, 1) == 15);
    CHECK(X(0, 2) == 16);
    CHECK(X(1, 0) == 24);
    CHECK(X(1, 1) == 25);
    CHECK(X(1, 2) == 26);
    zipper::Matrix<double, 2, 3> B;
    CHECK(B.extents() == zipper::create_dextents(2, 3));
    CHECK(X.extents() == zipper::create_dextents(2, 3));
    B.row(0) = {0, 1, 2};
    B.row(1) = {3, 4, 5};
    X = B;
    CHECK(X.extents() == zipper::create_dextents(2, 3));
    CHECK((X == B));
    zipper::Matrix<double, 2, 4> C;
    CHECK(C.extents() == zipper::create_dextents(2, 4));
    C.row(0) = {0, 1, 2, 8};
    C.row(1) = {3, 4, 5, 8};
    X = C;
    CHECK(X.extents() == zipper::create_dextents(2, 4));
    CHECK((X == C));
  }
}
