import react from "react"
import NavigationBar from "@/components/navbar"
import { Montserrat } from "next/font/google"

const montserratFont = Montserrat({
    subsets : ["latin"],
    weight : "100"
})


export default function AboutUsPage() {
    return (
        <div className={`${montserratFont.className}`}>
            <div className="bg-white h-screen w-auto text-black relative">
                <div>
                    <NavigationBar/>
                </div>

                <div className="p-10 flex flex-col gap-y-14">
                    <div className="text-8xl">
                        <h1 className="text-slate-700 text-extralight">Tim Kami</h1>
                    </div> 

                    <div className="font-normal text-slate-800 text-4xl flex flex-col justify-between gap-y-2">
                        <div className="flex flex-row justify-between">
                            <span><h1>Michael Yulianto Tamba</h1></span>
                            <span>01082240012</span>
                        </div>
                        <div className="flex flex-row justify-between">
                            <span><h1>Fronli Asian Samuel</h1></span>
                            <span> 01082240018</span>
                        </div>
                        <div className="flex flex-row justify-between">
                            <span><h1>Darren Marvel</h1></span>
                            <span>01112240014</span>
                        </div>
                        <div className="flex flex-row justify-between">
                            <span><h1>Benaya Simamora</h1></span>
                            <span>01082240013</span>
                        </div>
                    </div>


                    <div className="flex flex-col font-light text-slate-800 absolute right-10 bottom-10" >
                        <span className="text-2xl">Project Matakuliah Sistem Basis Data</span>
                        <span className="text-xl">Universitas Pelita Harapan, Lippo Village</span>
                    </div>
                </div>
            </div>
        </div>
    );
}
