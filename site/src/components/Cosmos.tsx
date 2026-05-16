/**
 * Decorative deep-space overlay: nebulae + shooting stars + orbital ring.
 * Lighter blur radii so headless captures don't choke on backdrop expense.
 */
export default function Cosmos() {
  return (
    <div
      aria-hidden
      className="fixed inset-0 pointer-events-none z-0 overflow-hidden"
    >
      {/* starfield — two pseudo layers via radial-gradient stacks */}
      <div
        className="absolute inset-0"
        style={{
          backgroundImage: [
            'radial-gradient(1.2px 1.2px at 10% 12%, #fff, transparent 60%)',
            'radial-gradient(1.2px 1.2px at 28% 67%, rgba(255,255,255,0.82), transparent 60%)',
            'radial-gradient(1.2px 1.2px at 47% 23%, #fff, transparent 60%)',
            'radial-gradient(1.2px 1.2px at 63% 78%, rgba(255,255,255,0.72), transparent 60%)',
            'radial-gradient(1.2px 1.2px at 82% 41%, #fff, transparent 60%)',
            'radial-gradient(1.2px 1.2px at 6% 88%, rgba(212,200,255,0.85), transparent 60%)',
            'radial-gradient(1.2px 1.2px at 19% 36%, rgba(255,255,255,0.65), transparent 60%)',
            'radial-gradient(1.2px 1.2px at 35% 91%, #fff, transparent 60%)',
            'radial-gradient(1.5px 1.5px at 51% 52%, rgba(176,124,240,0.95), transparent 60%)',
            'radial-gradient(1.2px 1.2px at 71% 17%, #fff, transparent 60%)',
            'radial-gradient(1.5px 1.5px at 88% 76%, rgba(0,201,167,0.95), transparent 60%)',
            'radial-gradient(1.2px 1.2px at 96% 4%, #fff, transparent 60%)',
            'radial-gradient(1.2px 1.2px at 3% 58%, #fff, transparent 60%)',
            'radial-gradient(1.2px 1.2px at 41% 5%, rgba(255,255,255,0.7), transparent 60%)',
            'radial-gradient(1.2px 1.2px at 78% 95%, #fff, transparent 60%)',
            'radial-gradient(1.2px 1.2px at 56% 81%, #fff, transparent 60%)',
            'radial-gradient(2px 2px at 22% 19%, #fff, transparent 65%)',
            'radial-gradient(2px 2px at 58% 14%, rgba(176,124,240,0.95), transparent 65%)',
            'radial-gradient(2px 2px at 33% 72%, #fff, transparent 65%)',
            'radial-gradient(2px 2px at 85% 64%, rgba(0,201,167,0.9), transparent 65%)',
            'radial-gradient(2px 2px at 70% 88%, #fff, transparent 65%)',
            'radial-gradient(2px 2px at 8% 28%, rgba(255,255,255,0.95), transparent 65%)',
          ].join(','),
          animation: 'twinkle 7s ease-in-out infinite',
        }}
      />

      {/* large purple nebula bloom — top-left */}
      <div
        className="absolute -top-32 -left-32 w-[560px] h-[560px] rounded-full opacity-70"
        style={{
          background:
            'radial-gradient(circle, rgba(176,124,240,0.4) 0%, rgba(132,94,194,0.18) 38%, transparent 70%)',
          filter: 'blur(24px)',
        }}
      />

      {/* teal accretion glow — mid-right */}
      <div
        className="absolute top-[40%] right-[-160px] w-[480px] h-[480px] rounded-full opacity-55"
        style={{
          background:
            'radial-gradient(circle, rgba(0,201,167,0.32) 0%, rgba(0,201,167,0.12) 40%, transparent 72%)',
          filter: 'blur(28px)',
        }}
      />

      {/* deep purple haze bottom-center */}
      <div
        className="absolute bottom-[-220px] left-1/4 w-[640px] h-[640px] rounded-full opacity-55"
        style={{
          background:
            'radial-gradient(circle, rgba(90,46,158,0.45) 0%, rgba(132,94,194,0.18) 42%, transparent 75%)',
          filter: 'blur(32px)',
        }}
      />

      {/* shooting stars */}
      <div
        className="absolute top-[20%] right-0 w-[180px] h-[2px] rounded-full"
        style={{
          background:
            'linear-gradient(90deg, transparent, rgba(176,124,240,0.95), rgba(255,255,255,1))',
          boxShadow: '0 0 8px rgba(176,124,240,0.9)',
          animation: 'shoot 14s linear infinite',
          animationDelay: '3s',
        }}
      />
      <div
        className="absolute top-[58%] right-[-50px] w-[150px] h-[2px] rounded-full"
        style={{
          background:
            'linear-gradient(90deg, transparent, rgba(0,201,167,0.9), rgba(255,255,255,1))',
          boxShadow: '0 0 8px rgba(0,201,167,0.7)',
          animation: 'shoot 22s linear infinite',
          animationDelay: '10s',
        }}
      />

      {/* faint orbital ring — distant planet/galaxy hint */}
      <div
        className="absolute top-[26%] right-[7%] w-[260px] h-[260px] rounded-full opacity-25"
        style={{
          border: '1px solid rgba(176,124,240,0.45)',
          boxShadow:
            '0 0 26px rgba(132,94,194,0.25), inset 0 0 26px rgba(132,94,194,0.15)',
        }}
      />
      <div
        className="absolute top-[26%] right-[7%] w-[260px] h-[260px] rounded-full opacity-20"
        style={{
          border: '1px dashed rgba(0,201,167,0.45)',
          transform: 'rotate(35deg)',
        }}
      />
    </div>
  )
}
