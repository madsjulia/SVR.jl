type Point{T}
  x::T
  y::T
end

function p(x, y)
  return Point(x, y)
end

function lattice(pi::Point{Real})
  x1, x2 = floor(pi.x), ceil(pi.x)
  y1, y2 = floor(pi.y), ceil(pi.y)
  return p(x1, y1), p(x1, y2), p(x2, y1), p(x2, y2)
end

function lattice(x::Real, y::Real)
  pi = p(x, y)
  return lattice(pi)
end

function drawLattice(pi::Point{Real})
  l1,l2,l3,l4 = lattice(pi)
  image = plot(x=[pi.x, l1.x, l2.x, l3.x, l4.x], y=[pi.y, l1.y, l2.y, l3.y, l4.y], Geom.point)
  draw(image)
end

function drawLattice(x::Real, y::Real)
  drawLattice(p(x, y))
end




