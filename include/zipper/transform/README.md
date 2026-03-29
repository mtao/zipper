# Transforms

Geometric transform types and factory functions for spatial transformations,
projections, and coordinate mapping. All types live in the `zipper::transform`
namespace and follow right-handed coordinate conventions (OpenGL style).

---

## Transform Types

The transform system provides both a general matrix-backed type and
lightweight specialised types that store only the minimal data required.

### General Transform

| Header | Type | Description |
|--------|------|-------------|
| `Transform.hpp` | `Transform<T, D, Mode>` | Primary matrix-backed transform owning a (D+1)x(D+1) matrix. Provides `linear()`, `translation()`, `matrix()`, `inverse()`. Default-constructs to identity. |

Convenience aliases:

- `AffineTransform<T, D>` = `Transform<T, D, TransformMode::Affine>`
- `Isometry<T, D>` = `Transform<T, D, TransformMode::Isometry>`
- `ProjectiveTransform<T, D>` = `Transform<T, D, TransformMode::Projective>`

`TransformMode` controls inverse computation and composition semantics:

| Mode | Assumption | Inverse Strategy |
|------|-----------|-----------------|
| `Projective` | No assumptions about last row | General matrix inverse |
| `Affine` | Last row is `[0...0 1]`, linear block may contain scale/shear | Block inverse (cheaper) |
| `Isometry` | Last row is `[0...0 1]`, linear block is orthogonal | Transpose of linear block (cheapest) |

When two transforms are composed via `operator*`, the result's mode is the
least restrictive (most general) of the two operands.

### Specialised Types

These store only the data they need and compose with each other and with
`Transform` via `operator*` (converting to matrix form when necessary).

| Header | Type | Storage | Description |
|--------|------|---------|-------------|
| `Rotation.hpp` | `Rotation<T, D>` | DxD orthogonal matrix | Pure rotation; inverse is transpose |
| `Scaling.hpp` | `Scaling<T, D>` | D-dimensional vector | Per-axis scale factors |
| `Translation.hpp` | `Translation<T, D>` | D-dimensional vector | Pure translation |
| `AxisAngleRotation.hpp` | `AxisAngleRotation<T>` | Angle + 3D axis vector | 3D axis-angle rotation; converts to `Rotation<T,3>` for composition |

`AffineTransform.hpp` is a backward-compatibility forwarding header that
includes `Transform.hpp`; new code should use `Transform.hpp` directly.

---

## Factory Functions

### Model Transforms

| Header | Function | Description |
|--------|----------|-------------|
| `model.hpp` | `translation(v)` | Standalone translation from a displacement vector |
| `model.hpp` | `rotation(angle, axis)` | Standalone rotation from angle (radians) and axis |
| `model.hpp` | `scaling(v)` | Standalone scaling from a scale vector |

### View Matrix

| Header | Function | Description |
|--------|----------|-------------|
| `view.hpp` | `look_at(eye, center, up)` | Right-handed view (camera) matrix |

### Projection Matrices

| Header | Function | Description |
|--------|----------|-------------|
| `projection.hpp` | `ortho(left, right, bottom, top)` | 2D orthographic projection |
| `projection.hpp` | `ortho(left, right, bottom, top, zNear, zFar)` | 3D orthographic projection (RH, depth [-1,1]) |
| `projection.hpp` | `frustum(left, right, bottom, top, zNear, zFar)` | Perspective from raw frustum planes (RH, depth [-1,1]) |
| `projection.hpp` | `perspective(fovy, aspect, zNear, zFar)` | Perspective from FOV and aspect ratio |

---

## Quaternion Transforms

| Header | Function | Description |
|--------|----------|-------------|
| `quaternion_transform.hpp` | `angle_axis(angle, axis)` | Create quaternion from axis-angle |
| `quaternion_transform.hpp` | `euler_angles(q)` | Extract (pitch, yaw, roll) from quaternion |
| `quaternion_transform.hpp` | `to_rotation_matrix(q)` | Quaternion to 3x3 rotation matrix |
| `quaternion_transform.hpp` | `to_affine(q)` | Quaternion to `AffineTransform` |
| `quaternion_transform.hpp` | `to_quaternion(M)` | 3x3 matrix or affine transform to quaternion |
| `quaternion_transform.hpp` | `slerp(q1, q2, t)` | Spherical linear interpolation |

---

## Geometric Utilities

| Header | Function | Description |
|--------|----------|-------------|
| `geometric.hpp` | `distance(p0, p1)` | Euclidean distance between two points |
| `geometric.hpp` | `reflect(I, N)` | Reflect incident vector about a normal |
| `geometric.hpp` | `refract(I, N, eta)` | Refract incident vector through a surface |
| `geometric.hpp` | `face_forward(N, I, Nref)` | Orient normal to face the viewer |
| `coordinate.hpp` | `project(obj, model, proj, viewport)` | Object-space to window-space (like `gluProject`) |
| `coordinate.hpp` | `unproject(win, model, proj, viewport)` | Window-space to object-space (like `gluUnProject`) |
| `common.hpp` | `radians(degrees)` | Degrees to radians |
| `common.hpp` | `degrees(radians)` | Radians to degrees |

---

## Transform Decomposition

| Header | Function | Description |
|--------|----------|-------------|
| `decompose.hpp` | `trs_decompose(xf)` | Extract Translation, Rotation, Scaling from an affine transform (assumes no shear) |

---

## Composition

| Header | Description |
|--------|-------------|
| `transform_compose.hpp` | Generic `operator*` overloads for composing any combination of transform types; converts specialised types to matrix form when required |

---

## Umbrella Header

| Header | Description |
|--------|-------------|
| `transform.hpp` | Includes all transform sub-headers for convenience |

---

## Implementation Detail

| Path | Purpose |
|------|---------|
| `detail/TransformBase.hpp` | `TransformMode` enum, `promote_mode()`, `IsTransform`/`IsMatrixTransform` traits, transform concepts, and `operator*` overloads for matrix-backed transforms |
| `detail/AffineTransformBase.hpp` | Internal base class and helpers for affine transform types |
