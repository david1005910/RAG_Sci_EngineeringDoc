import { useEffect, useState } from 'react'

interface DNAHelixProps {
  count?: number
}

export default function DNAHelix({ count = 3 }: DNAHelixProps) {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) return null

  const helixPositions = [
    { left: '3%', top: '5%', scale: 1.1, delay: 0, rotate: -15 },
    { right: '5%', top: '10%', scale: 0.95, delay: 2, rotate: 10 },
    { left: '8%', bottom: '5%', scale: 0.85, delay: 4, rotate: -20 },
    { right: '3%', bottom: '10%', scale: 0.75, delay: 1, rotate: 15 },
  ].slice(0, count)

  return (
    <>
      {helixPositions.map((pos, index) => (
        <div
          key={index}
          className="dna-helix-container"
          style={{
            position: 'absolute',
            left: pos.left,
            right: pos.right,
            top: pos.top,
            bottom: pos.bottom,
            transform: `scale(${pos.scale}) rotate(${pos.rotate}deg)`,
            opacity: 0.75,
            zIndex: 1,
          }}
        >
          <ProteinChemicalHelix delay={pos.delay} />
        </div>
      ))}
    </>
  )
}

interface ProteinChemicalHelixProps {
  delay: number
}

// CPK Coloring for atoms
const atomColors = {
  C: '#404040',   // Carbon - dark gray
  N: '#3050F8',   // Nitrogen - blue
  O: '#FF0D0D',   // Oxygen - red
  H: '#FFFFFF',   // Hydrogen - white
  R: '#00FF00',   // R group (side chain) - green
  S: '#FFFF30',   // Sulfur - yellow
}

const atomLabels = {
  C: 'C',
  N: 'N',
  O: 'O',
  H: 'H',
  R: 'R',
  S: 'S',
}

interface AtomProps {
  x: number
  y: number
  z: number
  type: keyof typeof atomColors
  size: number
  helixRadius: number
  delay: number
  index: number
  showLabel?: boolean
}

function Atom({ x, y, z, type, size, helixRadius, delay: _delay, index: _index, showLabel = true }: AtomProps) {
  // _delay and _index are passed for future animation support
  void _delay
  void _index
  const depthFactor = (z + helixRadius) / (helixRadius * 2)
  const opacity = 0.6 + depthFactor * 0.4
  const scale = 0.75 + depthFactor * 0.45
  const actualSize = size * scale

  // 3D Sphere colors with full gradient range
  const sphereColors: Record<string, { base: string; highlight: string; mid: string; shadow: string; rim: string }> = {
    C: { base: '#3a3a3a', highlight: '#888888', mid: '#4a4a4a', shadow: '#151515', rim: '#2a2a2a' },
    N: { base: '#2255dd', highlight: '#88aaff', mid: '#3366ee', shadow: '#112266', rim: '#1a44aa' },
    O: { base: '#ee2222', highlight: '#ff8888', mid: '#ff4444', shadow: '#881111', rim: '#cc1a1a' },
    H: { base: '#ffffff', highlight: '#ffffff', mid: '#f5f5f5', shadow: '#aaaaaa', rim: '#dddddd' },
    R: { base: '#22cc22', highlight: '#88ff88', mid: '#44dd44', shadow: '#116611', rim: '#1aaa1a' },
    S: { base: '#eeee22', highlight: '#ffff99', mid: '#ffff44', shadow: '#888811', rim: '#cccc1a' },
  }

  const colors = sphereColors[type] || sphereColors.C

  return (
    <div
      style={{
        position: 'absolute',
        left: `${x - actualSize / 2}px`,
        top: `${y - actualSize / 2}px`,
        width: `${actualSize}px`,
        height: `${actualSize}px`,
        borderRadius: '50%',
        opacity: opacity,
        zIndex: Math.round(z + helixRadius + 20),
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        // Perfect 3D sphere with multiple gradient layers
        background: `
          radial-gradient(circle at 30% 30%, ${colors.highlight} 0%, transparent 25%),
          radial-gradient(circle at 35% 35%, rgba(255,255,255,0.9) 0%, transparent 15%),
          radial-gradient(circle at 40% 40%, ${colors.mid} 0%, transparent 50%),
          radial-gradient(circle at 50% 50%, ${colors.base} 0%, ${colors.shadow} 100%)
        `,
        boxShadow: `
          inset -${actualSize * 0.15}px -${actualSize * 0.15}px ${actualSize * 0.4}px ${colors.shadow},
          inset ${actualSize * 0.08}px ${actualSize * 0.08}px ${actualSize * 0.15}px rgba(255,255,255,0.4),
          inset 0 -${actualSize * 0.05}px ${actualSize * 0.1}px rgba(255,255,255,0.1),
          ${actualSize * 0.08}px ${actualSize * 0.12}px ${actualSize * 0.25}px rgba(0,0,0,0.5),
          ${actualSize * 0.02}px ${actualSize * 0.03}px ${actualSize * 0.08}px rgba(0,0,0,0.3)
        `,
        border: `1px solid ${colors.rim}`,
      }}
    >
      {/* Primary highlight - bright spot */}
      <div
        style={{
          position: 'absolute',
          top: '12%',
          left: '18%',
          width: '30%',
          height: '22%',
          borderRadius: '50%',
          background: 'radial-gradient(ellipse at center, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.6) 30%, transparent 70%)',
          transform: 'rotate(-30deg)',
        }}
      />
      {/* Secondary highlight - softer reflection */}
      <div
        style={{
          position: 'absolute',
          top: '20%',
          left: '55%',
          width: '18%',
          height: '12%',
          borderRadius: '50%',
          background: 'radial-gradient(ellipse at center, rgba(255,255,255,0.4) 0%, transparent 70%)',
          transform: 'rotate(20deg)',
        }}
      />
      {/* Bottom rim light */}
      <div
        style={{
          position: 'absolute',
          bottom: '8%',
          left: '25%',
          width: '50%',
          height: '15%',
          borderRadius: '50%',
          background: `radial-gradient(ellipse at center, ${colors.rim}66 0%, transparent 70%)`,
        }}
      />
      {showLabel && actualSize > 16 && (
        <span
          style={{
            fontSize: `${Math.max(10, actualSize * 0.42)}px`,
            fontWeight: 'bold',
            color: type === 'H' || type === 'S' ? '#333' : '#fff',
            textShadow: type === 'H' || type === 'S'
              ? '0 1px 2px rgba(0,0,0,0.3), 0 0 4px rgba(255,255,255,0.5)'
              : '0 1px 3px rgba(0,0,0,0.9), 0 0 6px rgba(0,0,0,0.5)',
            fontFamily: 'Arial Black, Arial, sans-serif',
            letterSpacing: '-0.5px',
            zIndex: 1,
          }}
        >
          {atomLabels[type]}
        </span>
      )}
    </div>
  )
}

interface BondProps {
  x1: number
  y1: number
  z1: number
  x2: number
  y2: number
  z2: number
  helixRadius: number
  double?: boolean
}

function Bond({ x1, y1, z1, x2, y2, z2, helixRadius, double = false }: BondProps) {
  const avgZ = (z1 + z2) / 2
  const depthFactor = (avgZ + helixRadius) / (helixRadius * 2)
  const opacity = 0.3 + depthFactor * 0.5
  const strokeWidth = 2 + depthFactor * 2

  return (
    <svg
      style={{
        position: 'absolute',
        left: 0,
        top: 0,
        width: '100%',
        height: '100%',
        overflow: 'visible',
        zIndex: Math.round(avgZ + helixRadius),
      }}
    >
      <line
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke="#888"
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        opacity={opacity}
      />
      {double && (
        <line
          x1={x1 + 3}
          y1={y1}
          x2={x2 + 3}
          y2={y2}
          stroke="#888"
          strokeWidth={strokeWidth * 0.7}
          strokeLinecap="round"
          opacity={opacity * 0.8}
        />
      )}
    </svg>
  )
}

function ProteinChemicalHelix({ delay }: ProteinChemicalHelixProps) {
  // Helix parameters
  const aminoAcids = 8 // Number of amino acid residues
  const helixHeight = 620
  const helixRadius = 50 // Constant radius for the helix
  const containerWidth = helixRadius * 2 + 80
  const centerX = containerWidth / 2
  const residueHeight = helixHeight / aminoAcids

  // Peptide backbone: N - Cα - C(=O) - [N - Cα - C(=O)]...
  // Each amino acid contributes: N, H, Cα, R, C, O

  const atoms: Array<{
    x: number
    y: number
    z: number
    type: keyof typeof atomColors
    size: number
  }> = []

  const bonds: Array<{
    x1: number
    y1: number
    z1: number
    x2: number
    y2: number
    z2: number
    double?: boolean
  }> = []

  for (let i = 0; i < aminoAcids; i++) {
    // Base angle for this residue (100° per residue for alpha-helix)
    const baseAngle = (i * 100 * Math.PI) / 180
    const baseY = i * residueHeight + 30

    // Nitrogen (N) - amide nitrogen
    const nAngle = baseAngle
    const nX = centerX + Math.cos(nAngle) * helixRadius
    const nZ = Math.sin(nAngle) * helixRadius
    const nY = baseY
    atoms.push({ x: nX, y: nY, z: nZ, type: 'N', size: 22 })

    // Hydrogen on N (H)
    const hAngle = baseAngle + 0.3
    const hX = centerX + Math.cos(hAngle) * (helixRadius + 18)
    const hZ = Math.sin(hAngle) * (helixRadius + 18)
    const hY = baseY - 5
    atoms.push({ x: hX, y: hY, z: hZ, type: 'H', size: 14 })
    bonds.push({ x1: nX, y1: nY, z1: nZ, x2: hX, y2: hY, z2: hZ })

    // Alpha Carbon (Cα)
    const caAngle = baseAngle + 0.5
    const caX = centerX + Math.cos(caAngle) * helixRadius
    const caZ = Math.sin(caAngle) * helixRadius
    const caY = baseY + residueHeight * 0.25
    atoms.push({ x: caX, y: caY, z: caZ, type: 'C', size: 20 })
    bonds.push({ x1: nX, y1: nY, z1: nZ, x2: caX, y2: caY, z2: caZ })

    // R group (side chain) - pointing outward
    const rAngle = caAngle
    const rX = centerX + Math.cos(rAngle) * (helixRadius + 25)
    const rZ = Math.sin(rAngle) * (helixRadius + 25)
    const rY = caY + 5
    atoms.push({ x: rX, y: rY, z: rZ, type: 'R', size: 18 })
    bonds.push({ x1: caX, y1: caY, z1: caZ, x2: rX, y2: rY, z2: rZ })

    // Carbonyl Carbon (C)
    const cAngle = baseAngle + 1.0
    const cX = centerX + Math.cos(cAngle) * helixRadius
    const cZ = Math.sin(cAngle) * helixRadius
    const cY = baseY + residueHeight * 0.55
    atoms.push({ x: cX, y: cY, z: cZ, type: 'C', size: 20 })
    bonds.push({ x1: caX, y1: caY, z1: caZ, x2: cX, y2: cY, z2: cZ })

    // Carbonyl Oxygen (O) - double bond
    const oAngle = cAngle + 0.3
    const oX = centerX + Math.cos(oAngle) * (helixRadius + 20)
    const oZ = Math.sin(oAngle) * (helixRadius + 20)
    const oY = cY + 3
    atoms.push({ x: oX, y: oY, z: oZ, type: 'O', size: 20 })
    bonds.push({ x1: cX, y1: cY, z1: cZ, x2: oX, y2: oY, z2: oZ, double: true })

    // Peptide bond to next N (if not last residue)
    if (i < aminoAcids - 1) {
      const nextBaseAngle = ((i + 1) * 100 * Math.PI) / 180
      const nextNX = centerX + Math.cos(nextBaseAngle) * helixRadius
      const nextNZ = Math.sin(nextBaseAngle) * helixRadius
      const nextNY = (i + 1) * residueHeight + 30
      bonds.push({ x1: cX, y1: cY, z1: cZ, x2: nextNX, y2: nextNY, z2: nextNZ })
    }

    // Hydrogen bonds (i to i+4 in alpha-helix) - dashed lines
    if (i >= 4) {
      const prevBaseAngle = ((i - 4) * 100 * Math.PI) / 180
      const prevOAngle = prevBaseAngle + 1.3
      const prevOX = centerX + Math.cos(prevOAngle) * (helixRadius + 20)
      const prevOZ = Math.sin(prevOAngle) * (helixRadius + 20)
      const prevOY = (i - 4) * residueHeight + 30 + residueHeight * 0.55 + 3

      // This will be rendered as a dashed line (hydrogen bond)
      bonds.push({
        x1: hX, y1: hY, z1: hZ,
        x2: prevOX, y2: prevOY, z2: prevOZ,
        double: false // We'll handle this specially
      })
    }
  }

  return (
    <div
      className="protein-chemical-helix"
      style={{
        width: `${containerWidth}px`,
        height: `${helixHeight}px`,
        position: 'relative',
        transformStyle: 'preserve-3d',
        animation: `dna-spin 18s linear ${delay}s infinite`,
      }}
    >
      {/* Render bonds first (behind atoms) */}
      {bonds.map((bond, i) => (
        <Bond
          key={`bond-${i}`}
          x1={bond.x1}
          y1={bond.y1}
          z1={bond.z1}
          x2={bond.x2}
          y2={bond.y2}
          z2={bond.z2}
          helixRadius={helixRadius}
          double={bond.double}
        />
      ))}

      {/* Render atoms */}
      {atoms.map((atom, i) => (
        <Atom
          key={`atom-${i}`}
          x={atom.x}
          y={atom.y}
          z={atom.z}
          type={atom.type}
          size={atom.size}
          helixRadius={helixRadius}
          delay={delay}
          index={i}
          showLabel={true}
        />
      ))}

      {/* Legend */}
      <div
        style={{
          position: 'absolute',
          bottom: '-60px',
          left: '50%',
          transform: 'translateX(-50%)',
          display: 'flex',
          gap: '8px',
          fontSize: '10px',
          color: 'rgba(255,255,255,0.7)',
          whiteSpace: 'nowrap',
        }}
      >
        <span style={{ color: atomColors.C }}>●C</span>
        <span style={{ color: atomColors.N }}>●N</span>
        <span style={{ color: atomColors.O }}>●O</span>
        <span style={{ color: atomColors.H }}>●H</span>
        <span style={{ color: atomColors.R }}>●R</span>
      </div>
    </div>
  )
}
