import math

class Easing:
    @staticmethod
    def Linear(t):
        return t

    @staticmethod
    def SineEaseIn(t):
        return -math.cos(t * math.pi / 2) + 1

    @staticmethod
    def SineEaseOut(t):
        return math.sin(t * math.pi / 2)

    @staticmethod
    def SineEaseInOut(t):
        return -(math.cos(math.pi * t) - 1) / 2

    @staticmethod
    def QuadEaseIn(t):
        return t * t

    @staticmethod
    def QuadEaseOut(t):
        return -t * (t - 2)

    @staticmethod
    def QuadEaseInOut(t):
        t *= 2
        if t < 1:
            return t * t / 2
        else:
            t -= 1
            return -(t * (t - 2) - 1) / 2

    @staticmethod
    def CubicEaseIn(t):
        return t * t * t

    @staticmethod
    def CubicEaseOut(t):
        t -= 1
        return t * t * t + 1

    @staticmethod
    def CubicEaseInOut(t):
        t *= 2
        if t < 1:
            return t * t * t / 2
        else:
            t -= 2
            return (t * t * t + 2) / 2

    @staticmethod
    def QuartEaseIn(t):
        return t * t * t * t

    @staticmethod
    def QuartEaseOut(t):
        t -= 1
        return -(t * t * t * t - 1)

    @staticmethod
    def QuartEaseInOut(t):
        t *= 2
        if t < 1:
            return t * t * t * t / 2
        else:
            t -= 2
            return -(t * t * t * t - 2) / 2

    @staticmethod
    def QuintEaseIn(t):
        return t * t * t * t * t

    @staticmethod
    def QuintEaseOut(t):
        t -= 1
        return t * t * t * t * t + 1

    @staticmethod
    def QuintEaseInOut(t):
        t *= 2
        if t < 1:
            return t * t * t * t * t / 2
        else:
            t -= 2
            return (t * t * t * t * t + 2) / 2

    @staticmethod
    def ExponentialEaseIn(t):
        return math.pow(2, 10 * (t - 1))

    @staticmethod
    def ExponentialEaseOut(t):
        return -math.pow(2, -10 * t) + 1

    @staticmethod
    def ExponentialEaseInOut(t):
        t *= 2
        if t < 1:
            return math.pow(2, 10 * (t - 1)) / 2
        else:
            t -= 1
            return -math.pow(2, -10 * t) - 1

    @staticmethod
    def CircularEaseIn(t):
        return 1 - math.sqrt(1 - t * t)

    @staticmethod
    def CircularEaseOut(t):
        t -= 1
        return math.sqrt(1 - t * t)

    @staticmethod
    def CircularEaseInOut(t):
        t *= 2
        if t < 1:
            return -(math.sqrt(1 - t * t) - 1) / 2
        else:
            t -= 2
            return (math.sqrt(1 - t * t) + 1) / 2

    @staticmethod
    def BounceEaseIn(t: float) -> float:
        return 1 - Easing.BounceEaseOut(1 - t)

    @staticmethod
    def BounceEaseOut(t: float) -> float:
        if t < 4 / 11:
            return 121 * t * t / 16
        elif t < 8 / 11:
            return (363 / 40.0 * t * t) - (99 / 10.0 * t) + 17 / 5.0
        elif t < 9 / 10:
            return (4356 / 361.0 * t * t) - (35442 / 1805.0 * t) + 16061 / 1805.0
        return (54 / 5.0 * t * t) - (513 / 25.0 * t) + 268 / 25.0

    @staticmethod
    def BounceEaseInOut(t: float) -> float:
        if t < 0.5:
            return 0.5 * Easing.BounceEaseIn(t * 2)
        return 0.5 * Easing.BounceEaseOut(t * 2 - 1) + 0.5
    
    @staticmethod
    def ElasticEaseIn(t: float) -> float:
        return math.sin(13 * math.pi / 2 * t) * math.pow(2, 10 * (t - 1))


    @staticmethod
    def ElasticEaseOut(t: float) -> float:
        return math.sin(-13 * math.pi / 2 * (t + 1)) * math.pow(2, -10 * t) + 1


    @staticmethod
    def ElasticEaseInOut(t: float) -> float:
        if t < 0.5:
            return (
                0.5
                * math.sin(13 * math.pi / 2 * (2 * t))
                * math.pow(2, 10 * ((2 * t) - 1))
            )
        return 0.5 * (
            math.sin(-13 * math.pi / 2 * ((2 * t - 1) + 1))
            * math.pow(2, -10 * (2 * t - 1))
            + 2
        )

    @staticmethod
    def ease(easing_name: str, t: float) -> float:
        # Convert from "Title Case With Spaces" to single word
        easing_name = easing_name.replace(" ", "")

        if hasattr(Easing, easing_name):
            return getattr(Easing, easing_name)(t)
        return Easing.linear(t)

    @staticmethod
    def get_easing_functions() -> list[str]:
        names = [name for name in dir(Easing) if not name.startswith('_') and name not in ['ease', 'get_easing_functions']]
        # Sort names with easeInOutQuad and linear first
        
        # Convert camelCase to Title Case With Spaces
        names = [
            ''.join(' ' + c if c.isupper() else c for c in name).strip()
            for name in names
        ]

        # Sort by last word in name
        names.sort()


        return names
